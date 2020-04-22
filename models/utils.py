# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Utils for deep hedging models."""

import logging
import random

import numpy as np
import tensorflow as tf

from plotting import ResultTypes, plot_heatmap, plot_deltas, plot_loss, plot_pnls

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def set_seed(seed=1):
    """Seed the RNGs for consistent results from run to run"""
    log.info('Using seed %d', seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
