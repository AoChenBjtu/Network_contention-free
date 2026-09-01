#!/usr/bin/python
from __future__ import division
from __future__ import print_function

"""Train LISTA at 20 dB and test the same network at 10 dB.

The network architecture, training problem, training schedule, sensing matrix,
and checkpoint are identical to ``LISTA.py``.  Only the held-out test data are
regenerated at a different SNR immediately before ``do_testing``.
"""
import math
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tools import networks, problems, train


TRAIN_SNR_DB = 20
TEST_SNR_DB = 10
TEST_SEED = 10


np.random.seed(1)
tf.set_random_seed(1)


def set_shifted_test_data(prob, sample_count, snr_db, seed=TEST_SEED):
    """Replace only ``xval/yval`` using the existing sensing matrix.

    Keeping ``prob.A`` unchanged is essential: this experiment measures an SNR
    distribution shift, rather than a simultaneous sensing-matrix shift.
    """
    rng = np.random.RandomState(seed)
    M, N = prob.A.shape
    pnz = prob.pnz

    x_test = ((rng.uniform(0.0, 1.0, (N, sample_count)) < pnz)
              * rng.normal(0.0, 1.0, (N, sample_count)))
    x_test = x_test.astype(np.float32)

    noise_var = pnz * N / M * math.pow(10.0, -snr_db / 10.0)
    noise = rng.normal(
        0.0, math.sqrt(noise_var), (M, sample_count)).astype(np.float32)
    y_test = (np.matmul(prob.A, x_test) + noise).astype(np.float32)

    prob.xval = x_test
    prob.yval = y_test
    prob.SNR = snr_db
    prob.noise_var = np.array(noise_var, dtype=np.float32)
    return noise_var


def main():
    # This training problem is intentionally identical to LISTA.py.
    prob = problems.bernoulli_gaussian_trial(
        M=100, N=200, L=64, pnz=.1, kappa=None, SNR=TRAIN_SNR_DB)
    layers = networks.build_LISTA(
        prob, T=15, initial_lambda=.1, untied=False)
    training_stages = train.setup_training(
        layers, prob, trinit=1e-4, refinements=(.5, .1, .01))

    # Reuse LISTA.py's checkpoint.  If training is already complete, the
    # shared trainer restores it and skips the completed training stages.
    savefile = 'LISTA.npz'
    sess = train.do_training(training_stages, prob, savefile)
    sess.close()

    noise_var = set_shifted_test_data(
        prob, sample_count=64, snr_db=TEST_SNR_DB)
    print('LISTA distribution-shift test: train SNR={train_snr} dB, '
          'test SNR={test_snr} dB, noise variance={noise_var:.6f}'.format(
              train_snr=TRAIN_SNR_DB, test_snr=TEST_SNR_DB,
              noise_var=noise_var))

    nmse_dB, mu_user = train.do_testing(
        training_stages, prob, savefile)
    print('LISTA SNR-shift test nmse={nmse:.6f} dB, mu={mu:.6f}'.format(
        nmse=nmse_dB, mu=mu_user))


if __name__ == '__main__':
    main()
