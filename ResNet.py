#!/usr/bin/python
from __future__ import division
from __future__ import print_function

"""
Train and test a 15-stage fully connected ResNet for sparse recovery.

This is the stronger black-box AI baseline requested in the review response.
It uses a residual MLP instead of an image-oriented convolutional ResNet and
learns the mapping from ``prob.y_`` directly to ``prob.x_`` without using the
sensing matrix ``prob.A`` or ``prob.A_`` in its forward pass.

The experiment uses the same depth convention as DNN, LISTA, and LAMP:
``T=1`` is the initial learned estimate and every later T adds one complete
two-FC residual block.  Thus ``T=15`` contains fourteen residual blocks, so
``ResNet T=k`` and ``DNN T=k`` have comparable learned update depth.

Each residual block uses a 768-to-1280-to-768 feature transformation.  The
slightly narrower branch offsets the extra feedback inputs and keeps the total
parameter count within about 1.5% of the DNN baseline.  To prevent deep blocks
from losing the observation, every block receives the normalized measurement
and the preceding estimate in addition to the hidden state.  A gated candidate
estimate is blended with the preceding estimate by a bounded learnable step.
This makes identity propagation explicit while giving every stage a useful,
non-vanishing refinement path.

The ResNet-specific training setup retains the same greedy extension and
end-to-end refinement pattern as the other experiments, but its refinement
loss also deep-supervises earlier outputs and penalizes a new stage when its
per-sample squared error fails to improve on that of the preceding stage.  This
prevents late end-to-end updates from sacrificing shallow-stage accuracy merely
to improve the final output, and continues to reward useful late-stage updates.
Tensor layout follows the rest of the project: features are rows and samples
are columns, so the estimate has shape ``[N, batch_size]``.
"""
import csv
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tools import problems, train


np.random.seed(1)
tf.set_random_seed(1)


def _resnet_activation(x_, activation, leaky_alpha=0.1):
    """Apply an activation supported by the TensorFlow 1.x experiment code."""
    if activation is None or activation == 'linear':
        return x_
    if activation == 'relu':
        return tf.nn.relu(x_)
    if activation == 'leaky_relu':
        return tf.maximum(leaky_alpha * x_, x_)
    if activation == 'tanh':
        return tf.tanh(x_)
    raise ValueError('Unrecognized ResNet activation %s' % activation)


def _resnet_weight(shape, name, scale=1.0):
    """Create a variance-preserving weight in [fan_out, fan_in] layout."""
    fan_out, fan_in = shape
    std = scale * np.sqrt(2.0 / float(fan_in + fan_out))
    value = np.random.normal(size=shape).astype(np.float32) * std
    return tf.Variable(value, dtype=tf.float32, name=name)


def _resnet_bias(dim, name):
    return tf.Variable(np.zeros((dim, 1), dtype=np.float32),
                       dtype=tf.float32, name=name)


def _resnet_layer_norm(x_, dim, name, eps=1e-6):
    """Normalize each sample over its feature dimension.

    Layer normalization is used instead of batch normalization because the
    existing TensorFlow 1.x trainer has no training/evaluation mode switch or
    batch-normalization update operations.
    """
    mean_ = tf.reduce_mean(x_, axis=0, keep_dims=True)
    var_ = tf.reduce_mean(tf.square(x_ - mean_), axis=0, keep_dims=True)
    gamma_ = tf.Variable(np.ones((dim, 1), dtype=np.float32),
                         dtype=tf.float32, name=name + '_gamma')
    beta_ = tf.Variable(np.zeros((dim, 1), dtype=np.float32),
                        dtype=tf.float32, name=name + '_beta')
    normalized_ = gamma_ * (x_ - mean_) / tf.sqrt(var_ + eps) + beta_
    return normalized_, (gamma_, beta_)


def _resnet_rms_normalize(x_, eps=1e-6):
    """RMS-normalize each sample without adding trainable parameters."""
    scale_ = tf.sqrt(tf.reduce_mean(tf.square(x_), axis=0,
                                   keep_dims=True) + eps)
    return x_ / scale_


def _resnet_prediction_head(features_, feature_size, out_size, name,
                            gated=True, gate_pnz=0.1, gate_floor=0.20):
    """Create a signed value head with an optional learned sparsity gate."""
    W_value_ = _resnet_weight(
        (out_size, feature_size), name + '_W_value')
    b_value_ = _resnet_bias(out_size, name + '_b_value')
    value_ = tf.matmul(W_value_, features_) + b_value_
    if not gated:
        return value_, (W_value_, b_value_)

    W_gate_ = _resnet_weight(
        (out_size, feature_size), name + '_W_gate')
    # A very sparse initial gate saturates its sigmoid and starves the value
    # branch of gradients.  A modest floor still expresses the sparse prior
    # while keeping the head trainable from the first stage.
    gate_init = np.maximum(gate_pnz, 0.20)
    gate_init = np.minimum(np.maximum(gate_init, 1e-3), 1.0 - 1e-3)
    gate_bias = np.log(gate_init / (1.0 - gate_init))
    b_gate_ = tf.Variable(
        (gate_bias * np.ones((out_size, 1))).astype(np.float32),
        dtype=tf.float32, name=name + '_b_gate')
    gate_ = tf.sigmoid(tf.matmul(W_gate_, features_) + b_gate_)
    xhat_ = value_ * (gate_floor + (1.0 - gate_floor) * gate_)
    return xhat_, (W_value_, b_value_, W_gate_, b_gate_)


def _resnet_blend_estimate(xhat_, candidate_, layer_index,
                           initial_step=0.50):
    """Convexly blend a new full estimate with the preceding estimate."""
    if xhat_ is None:
        return candidate_, ()
    step_logit = np.log(initial_step / (1.0 - initial_step))
    step_logit_ = tf.Variable(
        np.array(step_logit, dtype=np.float32), dtype=tf.float32,
        name='ResNet_blend_logit_{0:02d}'.format(layer_index))
    step_ = tf.sigmoid(step_logit_)
    return xhat_ + step_ * (candidate_ - xhat_), (step_logit_,)


def build_ResNet(prob, T=15, hidden_size=768, ffn_size=1280,
                 activation='leaky_relu', gated_output=True):
    """Build a layer-wise supervised fully connected residual receiver.

    The first stage embeds the measurement in a fixed-width feature space and
    predicts x.  Every subsequent stage applies a complete two-FC residual
    block.  Its residual branch sees the hidden state, normalized measurement,
    and preceding x estimate; its readout predicts a new full candidate and
    blends it with the preceding estimate.  This matches the T-stage convention
    used by ``build_DNN`` while retaining a stronger information path.

    Args:
        prob: Problem object containing ``y_`` and ``x_`` placeholders.
        T: Number of supervised residual stages; the experiment uses 15.
        hidden_size: Feature width; set to 768 to match the DNN experiment.
        ffn_size: Residual-branch width; 1280 keeps the parameter budget close
            to the DNN after adding measurement/estimate feedback inputs.
        activation: ``relu``, ``leaky_relu``, ``tanh``, or ``linear``.
        gated_output: Whether to use the same learned sparse-output gate as DNN.

    Returns:
        A list of ``(name, xhat_tensor, new_variables)`` tuples for
        ``setup_resnet_training``.  ``T=15`` produces estimates named
        ``ResNet T=1`` through ``ResNet T=15``.
    """
    if T < 1:
        raise ValueError('T must be at least 1')
    if hidden_size < 1:
        raise ValueError('hidden_size must be at least 1')
    if ffn_size < 1:
        raise ValueError('ffn_size must be at least 1')

    M = prob.y_.get_shape().as_list()[0]
    N = prob.x_.get_shape().as_list()[0]
    if M is None or N is None:
        raise ValueError('ResNet requires statically known y/x feature sizes')
    gate_pnz = getattr(prob, 'pnz', 0.1)
    layers = []

    # T=1: condition the input while remaining completely independent of A.
    y_norm_, input_norm_vars = _resnet_layer_norm(
        prob.y_, M, 'ResNet_input_ln')
    y_rms_ = tf.sqrt(tf.reduce_mean(tf.square(prob.y_), axis=0,
                                   keep_dims=True) + 1e-6)
    W_input_ = _resnet_weight(
        (hidden_size, M), 'ResNet_input_W')
    W_input_raw_ = _resnet_weight(
        (hidden_size, M), 'ResNet_input_raw_W', scale=0.5)
    b_input_ = _resnet_bias(hidden_size, 'ResNet_input_b')
    # Layer normalization stabilizes optimization but discards the absolute
    # measurement scale.  The raw-y shortcut preserves that information.
    h_ = (tf.matmul(W_input_, y_norm_)
          + tf.matmul(W_input_raw_, prob.y_) + b_input_)
    h_, hidden_norm_vars = _resnet_layer_norm(
        h_, hidden_size, 'ResNet_hidden_ln')
    h_ = _resnet_activation(h_, activation)
    readout_ = tf.concat((h_, y_norm_, y_rms_), axis=0)
    candidate_, head_vars = _resnet_prediction_head(
        readout_, hidden_size + M + 1, N, 'ResNet_head_01',
        gated=gated_output, gate_pnz=gate_pnz)
    xhat_, blend_vars = _resnet_blend_estimate(None, candidate_, 1)
    layers.append(('ResNet T=1', xhat_,
                   input_norm_vars + (W_input_, W_input_raw_, b_input_)
                   + hidden_norm_vars + head_vars + blend_vars))

    # T=2..T: every reported stage adds one complete residual block.
    for stage in range(2, T + 1):
        block = stage - 1
        h_norm_, norm1_vars = _resnet_layer_norm(
            h_, hidden_size, 'ResNet_block_{0}_ln1'.format(block))

        # Re-inject y and the previous estimate at every depth.  The latter is
        # RMS-normalized to keep its scale comparable across early/late stages.
        # No sensing matrix or model-derived operation is used here.
        x_feedback_ = _resnet_rms_normalize(xhat_)
        block_input_ = tf.concat(
            (h_norm_, y_norm_, y_rms_, x_feedback_), axis=0)
        block_input_size = hidden_size + M + 1 + N
        W1_ = _resnet_weight(
            (ffn_size, block_input_size),
            'ResNet_block_{0}_W1'.format(block))
        b1_ = _resnet_bias(ffn_size,
                           'ResNet_block_{0}_b1'.format(block))
        residual_ = _resnet_activation(
            tf.matmul(W1_, block_input_) + b1_, activation)

        residual_norm_, norm2_vars = _resnet_layer_norm(
            residual_, ffn_size,
            'ResNet_block_{0}_ln2'.format(block))

        W2_ = _resnet_weight(
            (hidden_size, ffn_size),
            'ResNet_block_{0}_W2'.format(block))
        b2_ = _resnet_bias(
            hidden_size, 'ResNet_block_{0}_b2'.format(block))
        residual_ = tf.matmul(
            W2_, _resnet_activation(residual_norm_, activation)) + b2_
        initial_residual_scale = 0.25
        alpha_init = np.arctanh(initial_residual_scale)
        alpha_logit_ = tf.Variable(
            np.array(alpha_init, dtype=np.float32), dtype=tf.float32,
            name='ResNet_block_{0}_res_scale_logit'.format(block))
        alpha_ = tf.tanh(alpha_logit_)
        # Preserve an exact identity shortcut.  Activation is kept inside the
        # branch so negative information is not repeatedly clipped on the skip.
        h_ = h_ + alpha_ * residual_

        readout_ = tf.concat((h_, y_norm_, y_rms_), axis=0)
        candidate_, head_vars = _resnet_prediction_head(
            readout_, hidden_size + M + 1, N,
            'ResNet_head_{0:02d}'.format(stage),
            gated=gated_output, gate_pnz=gate_pnz)
        xhat_, blend_vars = _resnet_blend_estimate(
            xhat_, candidate_, stage, initial_step=0.50)
        block_vars = (norm1_vars + (W1_, b1_) + norm2_vars
                      + (W2_, b2_, alpha_logit_) + head_vars + blend_vars)
        layers.append(('ResNet T={0}'.format(stage), xhat_, block_vars))

    return layers


def ResNet(prob, T=15, hidden_size=768, ffn_size=1280,
           activation='leaky_relu', gated_output=True):
    """Alias matching the builder naming styles used in ``tools.networks``."""
    return build_ResNet(prob, T=T, hidden_size=hidden_size,
                        ffn_size=ffn_size,
                        activation=activation,
                        gated_output=gated_output)


def _resnet_metrics(prob, xhat_):
    """Build the same NMSE and active-user metric used by ``tools.train``."""
    maskX_ = getattr(prob, 'maskX_', 1)
    x_ = prob.x_ * maskX_
    xhat_masked_ = xhat_ * maskX_
    nmse_ = tf.nn.l2_loss(xhat_masked_ - x_) / tf.nn.l2_loss(x_)

    opts = getattr(prob, 'opts', None)
    nrows = prob.x_.get_shape().as_list()[0]
    nusers = (opts['C'] * opts['Nu']
              if opts is not None and 'C' in opts and 'Nu' in opts else None)
    nr = opts['Nr'] if opts is not None and 'Nr' in opts else None
    if nusers is not None and nr is not None and nrows == 2 * nusers:
        x_user_ = tf.concat(
            (tf.reshape(x_[:nusers, :], (nusers, -1, nr)),
             tf.reshape(x_[nusers:, :], (nusers, -1, nr))), axis=2)
        xhat_user_ = tf.concat(
            (tf.reshape(xhat_masked_[:nusers, :], (nusers, -1, nr)),
             tf.reshape(xhat_masked_[nusers:, :], (nusers, -1, nr))), axis=2)
    elif nusers is not None and nr is not None and nrows == nusers:
        x_user_ = tf.reshape(x_, (nusers, -1, nr))
        xhat_user_ = tf.reshape(xhat_masked_, (nusers, -1, nr))
    else:
        x_user_ = x_
        xhat_user_ = xhat_masked_

    user_error_ = tf.reduce_sum(tf.square(xhat_user_ - x_user_), axis=-1)
    user_signal_ = tf.reduce_sum(tf.square(x_user_), axis=-1)
    active_user_ = tf.greater(user_signal_, 1e-12)
    user_nmse_ = user_error_ / tf.maximum(user_signal_, 1e-12)
    detected_ = tf.logical_and(
        active_user_, tf.less_equal(user_nmse_, 10 ** (-15 / 10)))
    mu_detected_ = tf.reduce_sum(tf.cast(detected_, tf.float32))
    mu_real_ = tf.reduce_sum(tf.cast(active_user_, tf.float32))
    mu_user_ = mu_detected_ / tf.maximum(mu_real_, 1.0)
    return nmse_, mu_user_


def setup_resnet_training(layer_info, prob, trinit=2e-4,
                          refinements=(.5, .1, .01),
                          deep_supervision=0.20,
                          monotonic_weight=0.50,
                          improvement_margin=0.01):
    """Create ResNet training stages with stable end-to-end supervision.

    The returned tuple format is identical to ``train.setup_training``, so the
    common training and checkpoint code remains unchanged.  Existing LISTA,
    LAMP, and DNN code paths are not touched.

    ``deep_supervision`` retains an averaged loss on all preceding outputs
    during full-network refinements.  ``monotonic_weight`` penalizes examples
    that fail to reduce the preceding stage's squared error by the requested
    ``improvement_margin``.  The reported NMSE always uses the actual
    current-stage output; it is never clipped or replaced by a previous result.
    """
    if not 0 <= deep_supervision <= 1:
        raise ValueError('deep_supervision must be in [0, 1]')
    if monotonic_weight < 0:
        raise ValueError('monotonic_weight must be non-negative')
    if not 0 <= improvement_margin < 1:
        raise ValueError('improvement_margin must be in [0, 1)')
    if len(refinements):
        refinement_array = np.asarray(refinements)
        if refinement_array.min() <= 0 or refinement_array.max() > 1:
            raise ValueError('all refinements must be in (0, 1]')

    learning_rate_ = tf.Variable(
        trinit, name='ResNet_learning_rate', trainable=False)
    training_stages = []
    preceding_xhats = []
    preceding_losses = []

    for name, xhat_, var_list in layer_info:
        current_loss_ = tf.nn.l2_loss(xhat_ - prob.x_)
        training_loss_ = current_loss_

        if preceding_losses:
            auxiliary_loss_ = tf.add_n(preceding_losses) / float(
                len(preceding_losses))
            training_loss_ += deep_supervision * auxiliary_loss_

            current_sample_error_ = tf.reduce_sum(
                tf.square(xhat_ - prob.x_), axis=0)
            previous_sample_error_ = tf.reduce_sum(
                tf.square(preceding_xhats[-1] - prob.x_), axis=0)
            regression_penalty_ = 0.5 * tf.reduce_sum(tf.nn.relu(
                current_sample_error_
                - (1.0 - improvement_margin)
                * tf.stop_gradient(previous_sample_error_)))
            training_loss_ += monotonic_weight * regression_penalty_

        nmse_, mu_user_ = _resnet_metrics(prob, xhat_)
        train_ = tf.train.AdamOptimizer(learning_rate_).minimize(
            training_loss_, var_list=var_list)
        training_stages.append(
            (name, xhat_, training_loss_, nmse_, mu_user_, train_, var_list))

        for factor in refinements:
            refine_ = tf.train.AdamOptimizer(
                learning_rate_ * factor).minimize(training_loss_)
            training_stages.append(
                (name + ' trainrate=' + str(factor),
                 xhat_, training_loss_, nmse_, mu_user_, refine_, ()))

        preceding_xhats.append(xhat_)
        preceding_losses.append(current_loss_)

    return training_stages


def do_layerwise_testing(sess, training_stages, prob,
                         csvfile='ResNet15_layer_nmse.csv'):
    """Evaluate and save exactly one NMSE result for every ResNet depth."""
    layer_stages = [stage for stage in training_stages if len(stage[6])]
    rows = []
    for name, xhat_, loss_, nmse_, mu_user_, train_, var_list in layer_stages:
        nmse, mu_user = sess.run(
            [nmse_, mu_user_],
            feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})
        nmse_dB = 10 * np.log10(nmse)
        layer_number = int(name.split('=')[-1])
        rows.append((layer_number, nmse_dB, mu_user))
        print('{name}: nmse={nmse:.6f} dB, mu={mu:.6f}'.format(
            name=name, nmse=nmse_dB, mu=mu_user))

    with open(csvfile, 'w') as stream:
        writer = csv.writer(stream)
        writer.writerow(('method', 'layer_or_iteration', 'nmse_dB', 'mu_user'))
        for layer_number, nmse_dB, mu_user in rows:
            writer.writerow(('ResNet', layer_number, nmse_dB, mu_user))

    return rows[-1][1], rows[-1][2]


def main():
    prob = problems.bernoulli_gaussian_trial(
        M=100, N=200, L=64, pnz=.1, kappa=None, SNR=20)
    layers = build_ResNet(prob, T=15, hidden_size=768, ffn_size=1280,
                          activation='leaky_relu',
                          gated_output=True)

    training_stages = setup_resnet_training(
        layers, prob, trinit=2e-4, refinements=(.5, .1, .01),
        deep_supervision=0.20, monotonic_weight=0.50,
        improvement_margin=0.01)

    # Use a new filename: loading the previous architecture's ``done`` list
    # would otherwise skip every stage even though the graph has changed.
    savefile = 'ResNet15_feedback_deepsupervised_v2.npz'
    sess = train.do_training(training_stages, prob, savefile)
    nmse_dB, mu_user = do_layerwise_testing(
        sess, training_stages, prob,
        csvfile='ResNet15_layer_nmse.csv')
    sess.close()
    print('ResNet test nmse={nmse:.6f} dB, mu={mu:.6f}'.format(
        nmse=nmse_dB, mu=mu_user))


if __name__ == '__main__':
    main()
