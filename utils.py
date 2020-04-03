"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Mark Higgins, Ben Pryke
Description: Utilities for the deep hedging scripts.
"""

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def calc_expected_shortfall(pnls, pctile):
    """Calculate the expected shortfall across a number of paths pnls.
    
    Note: Conventionally, expected shortfall is often reported as a positive number but here
    we do not switch the sign.
    
    Parameters
    ----------
    pnls : :obj:`numpy.array`
        Array of pnls for a number of paths.
    """
    
    n_pct = int((100 - pctile) / 100 * len(pnls))
    pnls = np.sort(pnls)
    price = pnls[:n_pct].mean()
    
    return price


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
        log_spot += (mu - vol * vol / 2.) * dt + vol * rs
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
