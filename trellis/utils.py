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
    """Disables GPU in TensorFlow. Must be called before initialising TensorFlow."""
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
