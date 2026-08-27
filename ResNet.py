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

Each residual block uses the same 768-to-1536-to-768 feature expansion as the
DNN experiment.  A learned gated readout, also matched to the DNN baseline,
encourages sparse outputs without inserting the sensing matrix into the model.
The network exposes an estimate after every residual stage and uses the same
greedy layer-wise extension and refinement schedule as LISTA, LAMP, and DNN.
Later estimates learn small residual corrections to earlier estimates, which
encourages a progressively decreasing layer-wise NMSE.
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
    """Create a He-initialized dense weight in [fan_out, fan_in] layout."""
    _, fan_in = shape
    std = scale * np.sqrt(2.0 / float(fan_in))
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


def _resnet_prediction_head(h_, hidden_size, out_size, name, gated=True,
                            gate_pnz=0.1, gate_floor=0.05):
    """Create a signed value head with an optional learned sparsity gate."""
    W_value_ = _resnet_weight(
        (out_size, hidden_size), name + '_W_value',
        scale=1.0 / np.sqrt(2.0))
    b_value_ = _resnet_bias(out_size, name + '_b_value')
    value_ = tf.matmul(W_value_, h_) + b_value_
    if not gated:
        return value_, (W_value_, b_value_)

    W_gate_ = _resnet_weight(
        (out_size, hidden_size), name + '_W_gate',
        scale=1.0 / np.sqrt(2.0))
    gate_pnz = np.minimum(np.maximum(gate_pnz, 1e-3), 1.0 - 1e-3)
    gate_bias = np.log(gate_pnz / (1.0 - gate_pnz))
    b_gate_ = tf.Variable(
        (gate_bias * np.ones((out_size, 1))).astype(np.float32),
        dtype=tf.float32, name=name + '_b_gate')
    gate_ = tf.sigmoid(tf.matmul(W_gate_, h_) + b_gate_)
    xhat_ = value_ * (gate_floor + (1.0 - gate_floor) * gate_)
    return xhat_, (W_value_, b_value_, W_gate_, b_gate_)


def _resnet_update_estimate(xhat_, h_, hidden_size, out_size, layer_index,
                            gated, gate_pnz, correction_scale=0.1):
    """Predict x at one depth, refining the preceding estimate when present."""
    prediction_, head_vars = _resnet_prediction_head(
        h_, hidden_size, out_size,
        'ResNet_head_{0:02d}'.format(layer_index), gated=gated,
        gate_pnz=gate_pnz)
    if xhat_ is None:
        return prediction_, head_vars

    beta_ = tf.Variable(
        np.array(correction_scale, dtype=np.float32), dtype=tf.float32,
        name='ResNet_out_scale_{0:02d}'.format(layer_index))
    return xhat_ + beta_ * prediction_, head_vars + (beta_,)


def build_ResNet(prob, T=15, hidden_size=768, ffn_size=1536,
                 activation='leaky_relu', gated_output=True):
    """Build a layer-wise supervised fully connected residual receiver.

    The first stage embeds the measurement in a fixed-width feature space and
    predicts x.  Every subsequent stage applies a complete two-FC residual
    block and learns a residual correction to the preceding x estimate.  This
    matches the T-stage convention used by ``build_DNN``.

    Args:
        prob: Problem object containing ``y_`` and ``x_`` placeholders.
        T: Number of supervised residual stages; the experiment uses 15.
        hidden_size: Feature width; set to 768 to match the DNN experiment.
        ffn_size: Residual-branch expansion width; 1536 matches the DNN.
        activation: ``relu``, ``leaky_relu``, ``tanh``, or ``linear``.
        gated_output: Whether to use the same learned sparse-output gate as DNN.

    Returns:
        A list of ``(name, xhat_tensor, new_variables)`` tuples for
        ``train.setup_training``.  ``T=15`` produces estimates named
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

    # T=1: match the DNN input conditioning while remaining independent of A.
    y_norm_, input_norm_vars = _resnet_layer_norm(
        prob.y_, M, 'ResNet_input_ln')
    W_input_ = _resnet_weight(
        (hidden_size, M), 'ResNet_input_W')
    b_input_ = _resnet_bias(hidden_size, 'ResNet_input_b')
    h_ = tf.matmul(W_input_, y_norm_) + b_input_
    h_, hidden_norm_vars = _resnet_layer_norm(
        h_, hidden_size, 'ResNet_hidden_ln')
    h_ = _resnet_activation(h_, activation)
    xhat_, head_vars = _resnet_update_estimate(
        None, h_, hidden_size, N, 1, gated_output, gate_pnz)
    layers.append(('ResNet T=1', xhat_,
                   input_norm_vars + (W_input_, b_input_)
                   + hidden_norm_vars + head_vars))

    # T=2..T: every reported stage adds one complete residual block.
    for stage in range(2, T + 1):
        block = stage - 1
        h_norm_, norm1_vars = _resnet_layer_norm(
            h_, hidden_size, 'ResNet_block_{0}_ln1'.format(block))
        W1_ = _resnet_weight(
            (ffn_size, hidden_size),
            'ResNet_block_{0}_W1'.format(block))
        b1_ = _resnet_bias(ffn_size,
                           'ResNet_block_{0}_b1'.format(block))
        residual_ = _resnet_activation(
            tf.matmul(W1_, h_norm_) + b1_, activation)

        residual_norm_, norm2_vars = _resnet_layer_norm(
            residual_, ffn_size,
            'ResNet_block_{0}_ln2'.format(block))

        W2_ = _resnet_weight(
            (hidden_size, ffn_size),
            'ResNet_block_{0}_W2'.format(block),
            scale=0.25)
        b2_ = _resnet_bias(
            hidden_size, 'ResNet_block_{0}_b2'.format(block))
        residual_ = tf.matmul(
            W2_, _resnet_activation(residual_norm_, activation)) + b2_
        alpha_ = tf.Variable(
            np.array(0.1, dtype=np.float32), dtype=tf.float32,
            name='ResNet_block_{0}_res_scale'.format(block))
        # Standard post-activation residual shortcut.  This distinguishes the
        # ResNet block from the pre-activation residual FFN used by DNN.
        h_ = _resnet_activation(h_ + alpha_ * residual_, activation)

        xhat_, head_vars = _resnet_update_estimate(
            xhat_, h_, hidden_size, N, stage,
            gated_output, gate_pnz)
        block_vars = (norm1_vars + (W1_, b1_) + norm2_vars
                      + (W2_, b2_, alpha_) + head_vars)
        layers.append(('ResNet T={0}'.format(stage), xhat_, block_vars))

    return layers


def ResNet(prob, T=15, hidden_size=768, ffn_size=1536,
           activation='leaky_relu', gated_output=True):
    """Alias matching the builder naming styles used in ``tools.networks``."""
    return build_ResNet(prob, T=T, hidden_size=hidden_size,
                        ffn_size=ffn_size,
                        activation=activation,
                        gated_output=gated_output)


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
    layers = build_ResNet(prob, T=15, hidden_size=768, ffn_size=1536,
                          activation='leaky_relu',
                          gated_output=True)

    training_stages = train.setup_training(
        layers, prob, trinit=1e-4, refinements=(.5, .1, .01))

    savefile = 'ResNet15_stage_aligned_gated.npz'
    sess = train.do_training(training_stages, prob, savefile)
    nmse_dB, mu_user = do_layerwise_testing(
        sess, training_stages, prob,
        csvfile='ResNet15_layer_nmse.csv')
    sess.close()
    print('ResNet test nmse={nmse:.6f} dB, mu={mu:.6f}'.format(
        nmse=nmse_dB, mu=mu_user))


if __name__ == '__main__':
    main()
