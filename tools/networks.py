#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la

import tensorflow as tf
import tools.shrinkage as shrinkage


def _dnn_activation(x_, activation, leaky_alpha=0.1):
    if activation is None or activation == 'linear':
        return x_
    if activation == 'tanh':
        return tf.tanh(x_)
    if activation == 'relu':
        return tf.nn.relu(x_)
    if activation == 'leaky_relu':
        return tf.maximum(leaky_alpha * x_, x_)
    raise ValueError('Unrecognized DNN activation %s' % activation)


def _dnn_weight(shape, name, scale=1.0):
    fan_out, fan_in = shape
    std = np.sqrt(2.0 / float(fan_in + fan_out))
    return tf.Variable((scale * std * np.random.normal(size=shape)).astype(np.float32),
                       dtype=tf.float32, name=name)


def _dnn_bias(dim, name):
    return tf.Variable(np.zeros((dim, 1), dtype=np.float32), dtype=tf.float32, name=name)


def _dnn_layer_norm(x_, dim, name, eps=1e-6):
    mean_ = tf.reduce_mean(x_, axis=0, keep_dims=True)
    var_ = tf.reduce_mean(tf.square(x_ - mean_), axis=0, keep_dims=True)
    gamma_ = tf.Variable(np.ones((dim, 1), dtype=np.float32), dtype=tf.float32,
                         name=name + '_gamma')
    beta_ = tf.Variable(np.zeros((dim, 1), dtype=np.float32), dtype=tf.float32,
                        name=name + '_beta')
    y_ = gamma_ * (x_ - mean_) / tf.sqrt(var_ + eps) + beta_
    return y_, (gamma_, beta_)


def _dnn_prediction_head(h_, hidden_size, out_size, name, gated=True,
                         gate_pnz=0.1, gate_floor=0.05):
    Wv_ = _dnn_weight((out_size, hidden_size), name + '_Wv')
    bv_ = _dnn_bias(out_size, name + '_bv')
    value_ = tf.matmul(Wv_, h_) + bv_
    if not gated:
        return value_, (Wv_, bv_,)

    Wg_ = _dnn_weight((out_size, hidden_size), name + '_Wg')
    gate_pnz = np.minimum(np.maximum(gate_pnz, 1e-3), 1 - 1e-3)
    gate_bias = np.log(gate_pnz / (1.0 - gate_pnz))
    bg_ = tf.Variable((gate_bias * np.ones((out_size, 1))).astype(np.float32),
                      dtype=tf.float32, name=name + '_bg')
    gate_ = tf.sigmoid(tf.matmul(Wg_, h_) + bg_)
    return value_ * (gate_floor + (1.0 - gate_floor) * gate_), (Wv_, bv_, Wg_, bg_,)


def build_DNN(prob, T=15, hidden_size=768, ffn_size=1536,
              activation='leaky_relu', gated_output=True):
    """
    Builds a fully-connected residual DNN to infer x from prob.y_.

    The network uses only generic neural-network design: input normalization,
    dense FFN blocks, layer normalization, residual connections, nonlinear
    activations, and optional gated prediction heads. Layer 1 predicts x
    directly; later layers learn residual corrections to that prediction.
    """
    layers = []
    A = prob.A
    M, N = A.shape
    gate_pnz = getattr(prob, 'pnz', 0.1)

    y_, input_norm_vars = _dnn_layer_norm(prob.y_, M, 'DNN_input_ln')
    W_ = _dnn_weight((hidden_size, M), 'DNN_fc_0_W')
    b_ = _dnn_bias(hidden_size, 'DNN_fc_0_b')
    h_ = tf.matmul(W_, y_) + b_
    h_, norm_vars = _dnn_layer_norm(h_, hidden_size, 'DNN_ln_0')
    h_ = _dnn_activation(h_, activation)
    xhat_, head_vars = _dnn_prediction_head(h_, hidden_size, N, 'DNN_head_0',
                                            gated=gated_output,
                                            gate_pnz=gate_pnz)
    layers.append(('DNN T=1', xhat_,
                   input_norm_vars + (W_, b_) + norm_vars + head_vars))

    for t in range(1, T):
        hnorm_, norm1_vars = _dnn_layer_norm(h_, hidden_size,
                                             'DNN_block_{0}_ln1'.format(t))
        W1_ = _dnn_weight((ffn_size, hidden_size), 'DNN_block_{0}_W1'.format(t))
        b1_ = _dnn_bias(ffn_size, 'DNN_block_{0}_b1'.format(t))
        z_ = _dnn_activation(tf.matmul(W1_, hnorm_) + b1_, activation)
        z_, norm2_vars = _dnn_layer_norm(z_, ffn_size,
                                         'DNN_block_{0}_ln2'.format(t))
        W2_ = _dnn_weight((hidden_size, ffn_size),
                          'DNN_block_{0}_W2'.format(t), scale=0.25)
        b2_ = _dnn_bias(hidden_size, 'DNN_block_{0}_b2'.format(t))
        z_ = tf.matmul(W2_, _dnn_activation(z_, activation)) + b2_
        alpha_ = tf.Variable(np.array(0.1, dtype=np.float32),
                             dtype=tf.float32, name='DNN_res_scale_{0}'.format(t))
        h_ = h_ + alpha_ * z_

        delta_, head_vars = _dnn_prediction_head(h_, hidden_size, N,
                                                 'DNN_delta_head_{0}'.format(t),
                                                 gated=gated_output,
                                                 gate_pnz=gate_pnz)
        beta_ = tf.Variable(np.array(0.1, dtype=np.float32),
                            dtype=tf.float32, name='DNN_out_scale_{0}'.format(t))
        xhat_ = xhat_ + beta_ * delta_
        block_vars = norm1_vars + (W1_, b1_) + norm2_vars + (W2_, b2_, alpha_)
        layers.append(('DNN T=' + str(t + 1), xhat_,
                       block_vars + head_vars + (beta_,)))

    return layers


def DNN(prob, T=15, hidden_size=768, ffn_size=1536,
        activation='leaky_relu', gated_output=True):
    return build_DNN(prob, T=T, hidden_size=hidden_size, ffn_size=ffn_size,
                     activation=activation, gated_output=gated_output)


def build_SBL_GRU(prob, L, nd, T, untied=False):
    assert not untied, 'TODO: untied'
    layers = []
    A = prob.A
    M, N = A.shape
    prob.sigma = prob.noise_var.astype(np.float32)
    prob.sigma = prob.sigma.astype(np.float32)
    # initialize parameters(gamma, w, v, )
    A_rep = np.reshape(np.tile(A, [L, 1]), (L, M, N))  # L, M, N
    A_rep = A_rep.astype(np.float32)
    A_rept = A_rep.transpose(0, 2, 1)  # L, N, M
    Gamma = np.reshape(np.tile(np.eye(N), [L, 1]), (L, N, N))
    Gamma = Gamma.astype(np.float32)  # L, N, N
    Eye = np.reshape(np.tile(np.eye(M), [L, 1]), (L, M, M))
    Eye = Eye.astype(np.float32)

    AGamma = tf.matmul(A_rep, Gamma)  # L, M, N
    AGammaAT = tf.matmul(AGamma, A_rept)  # L, M, M
    Sigma_Y = tf.linalg.inv(prob.sigma * Eye + AGammaAT)  # L, M, M
    ATSigma_Y = tf.matmul(A_rept, Sigma_Y)  # L, N, M
    ATSigma_YA = tf.matmul(ATSigma_Y, A_rep)  # L, N, N
    w = tf.sqrt(tf.linalg.diag_part(ATSigma_YA[0, :, :]))
    w = tf.expand_dims(w, axis=0)
    for i in range(1, L):
        w0 = tf.sqrt(tf.linalg.diag_part(ATSigma_YA[i, :, :]))
        w0 = tf.expand_dims(w0, axis=0)
        w = tf.concat([w, w0], axis=0)  # L, N
    w = tf.tile(tf.expand_dims(w, axis=2), [1, 1, nd])  # L, N, 1
    GammaATSigma_Y = tf.matmul(Gamma, ATSigma_Y)  # L, N, M
    y_ = tf.tile(tf.expand_dims(prob.y_, axis=2), [1, 1, 1])
    y_ = tf.transpose(y_, perm=[1, 0, 2])
    u = tf.matmul(GammaATSigma_Y, y_)  # L, N, 1
    YsubAu = y_ - tf.matmul(A_rep, u)  # L, M, 1
    mu = 0.1
    v = u + mu * tf.matmul(A_rept, YsubAu)
    # initial_lambda = initial_lambda * np.ones((L, N, 1), dtype=np.float32)
    eta = shrinkage.joint_threshold_SBL_GRU
    gamma_hat = tf.multiply(tf.math.reciprocal(w), eta(tf.abs(v), 2 * prob.sigma * w))
    gamma = tf.multiply(tf.math.reciprocal(w), eta(tf.abs(v), 2 * prob.sigma * w))
    W_z = tf.Variable(0.5 * tf.ones(shape=(N, N + N), dtype=tf.float32), dtype=tf.float32, name='W_z')
    W_z_ = tf.reshape(tf.tile(W_z, [L, 1]), (L, N, N + N))

    for t in range(1, T):
        lam_ = tf.Variable(2 * prob.sigma, name='lam_{0}'.format(t))
        # Gate
        update_gate = tf.sigmoid(tf.matmul(W_z_, tf.concat([gamma, gamma_hat], axis=1))) - 0.5
        reset_gate = tf.math.reciprocal(w)  # L, N, 1
        gamma_hat = tf.multiply(reset_gate, eta(tf.abs(v), lam_ * w))  # L, N, 1
        gamma = tf.multiply(tf.ones(shape=(L, N, 1)) - update_gate, gamma_hat) + tf.multiply(update_gate, gamma)

        Gamma = tf.linalg.diag(gamma[0, :, 0])
        Gamma = tf.expand_dims(Gamma, axis=0)
        for i in range(1, L):
            Gamma0 = tf.linalg.diag(gamma[i, :, 0])
            Gamma0 = tf.expand_dims(Gamma0, axis=0)
            Gamma = tf.concat([Gamma, Gamma0], axis=0)
        A_ = tf.Variable(A, dtype=tf.float32, name='A_{0}'.format(t))
        A_rep_ = tf.reshape(tf.tile(A_, [L, 1]), (L, M, N))  # L, M, N
        AGamma = tf.matmul(A_rep_, Gamma)
        A_rept_ = tf.transpose(A_rep_, perm=[0, 2, 1])
        AGammaAT = tf.matmul(AGamma, A_rept_)
        # sigma_ = tf.Variable(prob.sigma, dtype=tf.float32, name='sigma_{0}'.format(t))
        Sigma_Y = tf.linalg.inv(prob.sigma * Eye + AGammaAT)  # L, M, M
        ATSigma_Y = tf.matmul(A_rept_, Sigma_Y)
        ATSigma_YA = tf.matmul(ATSigma_Y, A_rep_)  # L, N, N

        # parameters
        w = tf.sqrt(tf.linalg.diag_part(ATSigma_YA[0, :, :]))
        w = tf.expand_dims(w, axis=0)
        for i in range(1, L):
            w0 = tf.sqrt(tf.linalg.diag_part(ATSigma_YA[i, :, :]))
            w0 = tf.expand_dims(w0, axis=0)
            w = tf.concat([w, w0], axis=0)
        w = tf.tile(tf.expand_dims(w, axis=2), [1, 1, nd])
        GammaATSigma_Y = tf.matmul(Gamma, ATSigma_Y)
        y_ = tf.tile(tf.expand_dims(prob.y_, axis=2), [1, 1, 1])
        y_ = tf.transpose(y_, perm=[1, 0, 2])
        u = tf.matmul(GammaATSigma_Y, y_)
        u_hat = tf.transpose(tf.squeeze(u), perm=[1, 0])
        YsubAu = y_ - tf.matmul(A_rep_, u)
        v = u + mu * tf.matmul(A_rept_, YsubAu)  # L, N, 1

        # add layers
        layers.append(('SBL-GRU T=' + str(t), u_hat, (lam_, A_,)))
    return layers


def build_LISTA(prob, T, initial_lambda=.1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied, 'TODO: untied'
    eta = shrinkage.simple_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    B = A.T / (1.01 * la.norm(A, 2) ** 2)
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    S_ = tf.Variable(np.identity(N) - np.matmul(B, A), dtype=tf.float32, name='S_0')
    By_ = tf.matmul(B_, prob.y_)
    layers.append(('Linear', By_, None))

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob, 'iid', True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda * np.ones((N, 1), dtype=np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = eta(By_, lam0_)
    layers.append(('LISTA T=1', xhat_, (lam0_,)))
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        xhat_ = eta(tf.matmul(S_, xhat_) + By_, lam_)
        layers.append(('LISTA T=' + str(t + 1), xhat_, (lam_,)))
    return layers


def build_LAMP(prob, T, shrink, untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta, theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init=' + repr(theta_init))
    layers = []
    A = prob.A
    M, N = A.shape
    B = A.T / (1.01 * la.norm(A, 2) ** 2)
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    By_ = tf.matmul(B_, prob.y_)
    layers.append(('Linear', By_, None))

    if getattr(prob, 'iid', True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init * np.ones((N, 1), dtype=np.float32)
    theta_ = tf.Variable(theta_init, dtype=tf.float32, name='theta_0')
    OneOverM = tf.constant(float(1) / M, dtype=tf.float32)
    NOverM = tf.constant(float(N) / M, dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_), 0) * OneOverM
    (xhat_, dxdr_) = eta(By_, rvar_, theta_)
    layers.append(('LAMP-{0} T=1'.format(shrink), xhat_, (theta_,)))

    vt_ = prob.y_
    for t in range(1, T):
        if len(dxdr_.get_shape()) == 2:
            dxdr_ = tf.reduce_mean(dxdr_, axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul(prob.A_, xhat_) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_), 0) * OneOverM
        theta_ = tf.Variable(theta_init, name='theta_' + str(t))
        if untied:
            B_ = tf.Variable(B, dtype=tf.float32, name='B_' + str(t))
            rhat_ = xhat_ + tf.matmul(B_, vt_)
            layers.append(('LAMP-{0} linear T={1}'.format(shrink, t + 1), rhat_, (B_,)))
        else:
            rhat_ = xhat_ + tf.matmul(B_, vt_)

        (xhat_, dxdr_) = eta(rhat_, rvar_, theta_)
        layers.append(('LAMP-{0} non-linear T={1}'.format(shrink, t + 1), xhat_, (theta_,)))

    return layers


def build_LVAMP(prob, T, shrink):
    """
    Build the LVMAP network with an SVD parameterization.
    Learns the measurement noise variance and nonlinearity parameters
    """
    eta, theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init=' + repr(theta_init))
    layers = []
    A = prob.A
    M, N = A.shape
    AA = np.matmul(A, A.T)
    s2, U = la.eigh(AA)  # this is faster than svd, but less precise if ill-conditioned
    s = np.sqrt(s2)
    V = np.matmul(A.T, U) / s
    print('svd reconstruction error={nmse:.3f}dB'.format(nmse=20 * np.log10(la.norm(A - np.matmul(U * s, V.T)) / la.norm(A))))
    assert np.allclose(A, np.matmul(U * s, V.T), rtol=1e-4, atol=1e-4)
    V_ = tf.constant(V, dtype=tf.float32, name='V')

    # precompute some tensorflow constants
    rS2_ = tf.constant(np.reshape(1 / (s * s), (-1, 1)).astype(np.float32))  # reshape to (M,1) to allow broadcasting
    # rj_ = tf.zeros( (N,L) ,dtype=tf.float32)
    rj_ = tf.zeros_like(prob.x_)
    taurj_ = tf.reduce_sum(prob.y_ * prob.y_, 0) / N
    logyvar_ = tf.Variable(0.0, name='logyvar', dtype=tf.float32)
    yvar_ = tf.exp(logyvar_)
    ytilde_ = tf.matmul(tf.constant((U / s).T.astype(np.float32)), prob.y_)  # inv(S)*U*y
    Vt_ = tf.transpose(V_)

    xhat_ = tf.constant(0, dtype=tf.float32)
    for t in range(T):  # layers 0 thru T-1
        # linear step (LMMSE estimation and Onsager correction)
        varRat_ = tf.reshape(yvar_ / taurj_, (1, -1))  # one per column
        scale_each_ = 1 / (1 + rS2_ * varRat_)  # LMMSE scaling individualized per element {singular dimension,column}
        zetai_ = N / tf.reduce_sum(scale_each_, 0)  # one per column  (zetai_ is 1/(1-alphai) from Phil's derivation )
        adjust_ = (scale_each_ * (ytilde_ - tf.matmul(Vt_, rj_))) * zetai_  # adjustment in the s space
        ri_ = rj_ + tf.matmul(V_, adjust_)  # bring the adjustment back into the x space and apply it
        tauri_ = taurj_ * (zetai_ - 1)  # adjust the variance

        # non-linear step
        theta_ = tf.Variable(theta_init, dtype=tf.float32, name='theta_' + str(t))
        xhat_, dxdr_ = eta(ri_, tauri_, theta_)
        if t == 0:
            learnvars = None  # really means "all"
        else:
            learnvars = (theta_,)
        layers.append(('LVAMP-{0} T={1}'.format(shrink, t + 1), xhat_, learnvars))

        if len(dxdr_.get_shape()) == 2:
            dxdr_ = tf.reduce_mean(dxdr_, axis=0)
        zetaj_ = 1 / (1 - dxdr_)
        rj_ = (xhat_ - dxdr_ * ri_) * zetaj_  # apply Onsager correction
        taurj_ = tauri_ * (zetaj_ - 1)  # adjust the variance

    return layers


def build_LVAMP_dense(prob, T, shrink, iid=False):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta, theta_init = shrinkage.get_shrinkage_function(shrink)
    layers = []
    A = prob.A
    M, N = A.shape

    Hinit = np.matmul(prob.xinit, la.pinv(prob.yinit))
    H_ = tf.Variable(Hinit, dtype=tf.float32, name='H0')
    xhat_lin_ = tf.matmul(H_, prob.y_)
    layers.append(('Linear', xhat_lin_, None))

    if shrink == 'pwgrid':
        theta_init = np.linspace(.01, .99, 15).astype(np.float32)
    vs_def = np.array(1, dtype=np.float32)
    if not iid:
        theta_init = np.tile(theta_init, (N, 1, 1))
        vs_def = np.tile(vs_def, (N, 1))

    theta_ = tf.Variable(theta_init, name='theta0', dtype=tf.float32)
    vs_ = tf.Variable(vs_def, name='vs0', dtype=tf.float32)
    rhat_nl_ = xhat_lin_
    rvar_nl_ = vs_ * tf.reduce_sum(prob.y_ * prob.y_, 0) / N

    xhat_nl_, alpha_nl_ = eta(rhat_nl_, rvar_nl_, theta_)
    layers.append(('LVAMP-{0} T={1}'.format(shrink, 1), xhat_nl_, None))
    for t in range(1, T):
        alpha_nl_ = tf.reduce_mean(alpha_nl_, axis=0)  # each col average dxdr

        gain_nl_ = 1.0 / (1.0 - alpha_nl_)
        rhat_lin_ = gain_nl_ * (xhat_nl_ - alpha_nl_ * rhat_nl_)
        rvar_lin_ = rvar_nl_ * alpha_nl_ * gain_nl_

        H_ = tf.Variable(Hinit, dtype=tf.float32, name='H' + str(t))
        G_ = tf.Variable(.9 * np.identity(N), dtype=tf.float32, name='G' + str(t))
        xhat_lin_ = tf.matmul(H_, prob.y_) + tf.matmul(G_, rhat_lin_)

        layers.append(('LVAMP-{0} lin T='.format(shrink) + str(t + 1), xhat_lin_, (H_, G_,)))
        alpha_lin_ = tf.expand_dims(tf.diag_part(G_), 1)

        eps = .5 / N
        alpha_lin_ = tf.maximum(eps, tf.minimum(1 - eps, alpha_lin_))

        vs_ = tf.Variable(vs_def, name='vs' + str(t), dtype=tf.float32)

        gain_lin_ = vs_ * 1.0 / (1.0 - alpha_lin_)
        rhat_nl_ = gain_lin_ * (xhat_lin_ - alpha_lin_ * rhat_lin_)
        rvar_nl_ = rvar_lin_ * alpha_lin_ * gain_lin_

        theta_ = tf.Variable(theta_init, name='theta' + str(t), dtype=tf.float32)

        xhat_nl_, alpha_nl_ = eta(rhat_nl_, rvar_nl_, theta_)
        alpha_nl_ = tf.maximum(eps, tf.minimum(1 - eps, alpha_nl_))
        layers.append(('LVAMP-{0} T='.format(shrink) + str(t + 1), xhat_nl_, (vs_, theta_,)))

    return layers
