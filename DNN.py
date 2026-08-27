#!/usr/bin/python
from __future__ import division
from __future__ import print_function

"""
Train and test a 15-layer fully-connected DNN for sparse signal recovery.
"""
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

from tools import problems, networks, train


prob = problems.bernoulli_gaussian_trial(M=100, N=200, L=64, pnz=.1, kappa=None, SNR=20)

layers = networks.build_DNN(prob, T=15, hidden_size=768, ffn_size=1536,
                            activation='leaky_relu', gated_output=True)

training_stages = train.setup_training(layers, prob, trinit=1e-4, refinements=(.5, .1, .01))

sess = train.do_training(training_stages, prob, 'DNN_residual_ffn.npz')
nmse_dB, mu_user = train.do_testing(training_stages, prob, 'DNN_residual_ffn.npz')

print('DNN test nmse={nmse:.6f} dB, mu={mu:.6f}'.format(nmse=nmse_dB, mu=mu_user))
