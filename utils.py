# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Utilities for the deep hedging scripts."""

import logging
import time
import os
import sys

# Note: must not import `tensorflow` here as it is required that we do not by `disable_gpu`
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def disable_gpu():
    """Disables GPU in TensorFlow. Must be called before importing TensorFlow"""

    if 'tensorflow' in sys.modules:
        raise RuntimeError('disable_gpu imported after tensorflow')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def calc_expected_shortfall(pnls, pctile):
    """Calculate the expected shortfall across a number of paths pnls.
    
    Note: Conventionally, expected shortfall is often reported as a positive number, so
    here we switch the sign.
    
    Parameters
    ----------
    pnls : :obj:`numpy.array`
        Array of pnls for a number of paths.
    """

    n_pct = int((100 - pctile) / 100 * len(pnls))
    pnls = np.sort(pnls)
    price = -pnls[:n_pct].mean()

    return price


def get_progressive_min(array):
    """Returns an array representing the closest to zero so far in the given array.
    
    Specifically, output value at index i will equal `min(abs(array[:i+1]))`.
    
    Parameters
    ----------
    array : list of :obj:`~numbers.Number` or :obj:`numpy.array`
        Input
    
    Returns
    -------
    list
        Progressively "best so far" minimum values from the input array
    """

    result = [0] * len(array)
    best = abs(array[0])

    for i, value in enumerate(array):
        if abs(value) < abs(best):
            best = value

        result[i] = best

    return result


def generate_paths(n_paths=100, init_spot=1.0, n_steps=100, texp=1.0, vol=0.2, mu=0.0):
    """Generates spot curves using simple geometric Brownian motion.
    
    Parameters
    ----------
    n_curves : int
        The number of curves to generate
    init_spot : float
        Initial spot price
    n_steps : int
        Number of steps to simulate, the length of the curves
    texp : float
        Time to expiry, years
    vol : float
        Volatility
    mu : Expected upward drift per year, 0.08 = 8% per year
    
    Returns
    -------
    :obj:`numpy.array`
        Array of curves of size (`n_steps`, `n_curves`)
    """

    log_spot = np.zeros(n_paths)
    spot = np.zeros((n_steps, n_paths))
    init_spot = 1.0
    dt = texp / n_steps
    sqrtdt = dt ** 0.5

    for t in range(n_steps):
        rs = np.random.normal(0, sqrtdt, size=n_paths)
        log_spot += (mu - vol * vol / 2.0) * dt + vol * rs
        spot[t, :] = init_spot * np.exp(log_spot)

    log.info('Average final spot %.2f', np.mean(spot[n_paths, :]))

    return spot


def get_duration_desc(start):
    """Returns a string {min}:{sec} describing the duration since `start`
    
    Parameters
    ----------
    start : int or float
        Timestamp
    """

    end = time.time()
    duration = round(end - start)
    minutes, seconds = divmod(duration, 60)
    return '{:02d}:{:02d}'.format(minutes, seconds)
