#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf


def save_trainable_vars(sess, filename, **kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save = {}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename, **save)


def load_trainable_vars(sess, filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other = {}
    try:
        tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
        for k, d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign(tv[k], d))
            else:
                other[k] = d
    except IOError:
        pass
    return other


def setup_training(layer_info, prob, trinit=1e-3, refinements=(.5, .1, .01), final_refine=None):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_ = []
    nmse_ = []
    trainers_ = []
    assert np.array(refinements).min() > 0, 'all refinements must be in (0,1]'
    assert np.array(refinements).max() <= 1, 'all refinements must be in (0,1]'

    maskX_ = getattr(prob, 'maskX_', 1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ * maskX_)

    tr_ = tf.Variable(trinit, name='tr', trainable=False)
    training_stages = []
    for name, xhat_, var_list in layer_info:
        loss_ = tf.nn.l2_loss(xhat_ - prob.x_)
        nmse_ = tf.nn.l2_loss((xhat_ - prob.x_) * maskX_) / nmse_denom_
        x_for_mu_ = prob.x_ * maskX_
        xhat_for_mu_ = xhat_ * maskX_

        opts = getattr(prob, 'opts', None)
        nrows = prob.x_.get_shape().as_list()[0]
        nusers = opts['C'] * opts['Nu'] if opts is not None and 'C' in opts and 'Nu' in opts else None
        nr = opts['Nr'] if opts is not None and 'Nr' in opts else None
        if nusers is not None and nr is not None and nrows == 2 * nusers:
            x_user_ = tf.concat((tf.reshape(x_for_mu_[:nusers, :], (nusers, -1, nr)),
                                 tf.reshape(x_for_mu_[nusers:, :], (nusers, -1, nr))), axis=2)
            xhat_user_ = tf.concat((tf.reshape(xhat_for_mu_[:nusers, :], (nusers, -1, nr)),
                                    tf.reshape(xhat_for_mu_[nusers:, :], (nusers, -1, nr))), axis=2)
        elif nusers is not None and nr is not None and nrows == nusers:
            x_user_ = tf.reshape(x_for_mu_, (nusers, -1, nr))
            xhat_user_ = tf.reshape(xhat_for_mu_, (nusers, -1, nr))
        else:
            x_user_ = x_for_mu_
            xhat_user_ = xhat_for_mu_

        user_error_energy_ = tf.reduce_sum(tf.square(xhat_user_ - x_user_), axis=-1)
        user_signal_energy_ = tf.reduce_sum(tf.square(x_user_), axis=-1)
        active_user_ = tf.greater(user_signal_energy_, 1e-12)
        user_nmse_ = user_error_energy_ / tf.maximum(user_signal_energy_, 1e-12)
        mu_detected = tf.reduce_sum(tf.cast(tf.logical_and(active_user_, tf.less_equal(user_nmse_, 10 ** (-15 / 10))), tf.float32))
        mu_real = tf.reduce_sum(tf.cast(active_user_, tf.float32))
        mu_user_ = mu_detected / tf.maximum(mu_real, 1.0)
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append((name, xhat_, loss_, nmse_, mu_user_, train_, var_list))
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_ * fm).minimize(loss_)
            training_stages.append((name + ' trainrate=' + str(fm), xhat_, loss_, nmse_, mu_user_, train2_, ()))
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_ * final_refine).minimize(loss_)
        training_stages.append((name + ' final refine ' + str(final_refine), xhat_, loss_, nmse_, mu_user_, train2_, ()))

    return training_stages


def do_training(training_stages, prob, savefile, ivl=10, maxit=5000, better_wait=500):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)
    # nmse_log = []
    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, xhat_, loss_, nmse_, mu_user_, train_, var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])

        print(name + ' ' + describe_var_list)
        nmse_history = []
        for i in range(maxit + 1):
            if i % ivl == 0:
                nmse, mu_user = sess.run([nmse_, mu_user_], feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})  # Validation set
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history, nmse)
                nmse_dB = 10 * np.log10(nmse)
                nmsebest_dB = 10 * np.log10(nmse_history.min())
                sys.stdout.write(
                    '\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f}) mu={mu:.6f}'.format(i=i, nmse=nmse_dB, best=nmsebest_dB, mu=mu_user))
                sys.stdout.flush()
                if i % (100 * ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1  # how long ago was the best nmse?
                    if age_of_best * ivl > better_wait:
                        break  # if it has not improved on the best answer for quite some time, then move along
                # nmse_log = np.append(nmse_log, nmse_dB)
            y, x = prob(sess)
            sess.run(train_, feed_dict={prob.y_: y, prob.x_: x})
        done = np.append(done, name)

        log = log + '\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name, nmse=nmse_dB, i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess, savefile, **state)
    return sess


def do_testing(training_stages, prob, savefile, printfile='run.txt'):
    # The project targets TensorFlow 1.12/1.13, where Session and the global
    # initializer are top-level APIs.  tf.compat.v1 is not available in TF 1.12.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mu_user = 0
    nmse_dB = 0
    print('norms xtest:{xtest:.7f} ytest:{ytest:.7f}'.format(xtest=la.norm(prob.xval), ytest=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer
    for name, xhat_, loss_, nmse_, mu_user_, train_, var_list in training_stages:
        #        if name in done:
        #            print('Already did ' + name + '. Skipping.')
        #            continue
        #        if name != 'LISTA T=2 trainrate=0.001':
        #           print( name )
        #           continue
        x = prob.xval
        y = prob.yval
        xtilde, nmse, mu_user = sess.run([xhat_, nmse_, mu_user_], feed_dict={prob.y_: y, prob.x_: x})
        nmse_dB = 10 * np.log10(nmse)
        sys.stdout.write('\rnmse={nmse:.6f} dB, mu={mu:.6f}\n'.format(nmse=nmse_dB, mu=mu_user))
        sys.stdout.flush()
    sess.close()
    return nmse_dB, mu_user


def Accuracy(xtilde, x):
    l = xtilde.shape[1]
    x_t = np.where(x != 0, np.ones_like(x), np.zeros_like(x))
    k = np.sum(x_t, axis=0).astype(int)
    score_total_for_strict = 0
    score_total_for_loose = 0
    for i in range(l):
        score = 0
        label = np.argpartition(np.abs(x[:, i]), -k[i])[-k[i]:]
        pred = np.argpartition(np.abs(xtilde[:, i]), -k[i])[-k[i]:]
        for m in range(k[i]):
            for n in range(k[i]):
                score = score + (label[m] == pred[n])
        if score == k[i]:
            score_total_for_strict = score_total_for_strict + 1
        score_total_for_loose = score_total_for_loose + score / k[i]
    return score_total_for_strict / l, score_total_for_loose / l
