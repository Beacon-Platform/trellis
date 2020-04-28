# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Utils for deep hedging models."""

import logging
import random

import numpy as np
import tensorflow as tf

from plotting import ResultTypes, plot_heatmap, plot_deltas, plot_loss, plot_pnls
from utils import calc_expected_shortfall

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def set_seed(seed=1):
    """Seed the RNGs for consistent results from run to run.
    
    Parameters
    ----------
    seed : int
        RNG seed
    """
    
    log.info('Using seed %d', seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def estimate_expected_shortfalls(uh_pnls, bs_pnls, nn_pnls, pctile, *, verbose=1):
    """Estimate the unhedged, analytical, and model expected shortfalls from a simulation.
    
    These estimates are also estimates of the fair price of the instrument.
    
    Parameters
    ----------
    un_pnls : list of float or :obj:`numpy.array` of float
        Unhedged PNLs for n paths
    bs_pnls : list of float or :obj:`numpy.array` of float
        Black-Scholes analytical PNLs for n paths
    nn_pnls : list of float or :obj:`numpy.array` of float
        Neural network output PNLs for n paths
    pctile : int, float
        Percentage Expected Shortfall to calculate
    
    Returns
    -------
    tuple of float
        (unhedged ES, analytical ES, neural network ES)
    """
    
    uh_es = calc_expected_shortfall(uh_pnls, pctile)
    bs_es = calc_expected_shortfall(bs_pnls, pctile)
    nn_es = calc_expected_shortfall(nn_pnls, pctile)
    
    if verbose != 0:
        log.info('Unhedged ES      = % .5f (mean % .5f, std % .5f)', uh_es, np.mean(uh_pnls), np.std(uh_pnls))
        log.info('Deep hedging ES  = % .5f (mean % .5f, std % .5f)', nn_es, np.mean(nn_pnls), np.std(nn_pnls))
        log.info('Black-Scholes ES = % .5f (mean % .5f, std % .5f)', bs_es, np.mean(bs_pnls), np.std(bs_pnls))
    
    return uh_es, bs_es, nn_es
