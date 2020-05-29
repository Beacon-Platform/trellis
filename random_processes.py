# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Stochastic processes."""

import tensorflow as tf


@tf.function
def gbm(S0, mu, sigma, dt, n_steps, n_paths=1):
    """Simulates geometric Brownian motion (GBM).
    
    Parameters
    ----------
    S0 : float
        Initial value
    mu : float
        Expected upward drift per year, e.g. 0.08 = 8% per year
    sigma : float
        Volatility
    dt : float
        Length of each timestep in years, e.g. 1/12 = monthly
    n_steps : int
        Number of timesteps to simulate
    n_paths : int
        Number of paths to simulate
    
    Returns
    -------
    :obj:`~tensorflow.Tensor`
        2D array of paths, with shape `(n_steps + 1, n_paths)`, each starting at `S0`
    """

    S0 = tf.fill((1, n_paths), S0)
    w = tf.random.normal((n_steps, n_paths), 0, dt ** 0.5)
    log_spots = tf.math.cumsum((mu - sigma * sigma / 2.0) * dt + sigma * w)
    spots = S0 * tf.math.exp(log_spots)
    return tf.concat((S0, spots), axis=0)


@tf.function
def gbm2(S0, mu, sigma, dt, n_steps, rho):
    """Simulates correlated geometric Brownian motion (GBM) for 2 paths.
    
    Parameters
    ----------
    S0 : list, :obj:`~numpy.array`, or :obj:`~tensorflow.Tensor`
        Initial values, of length 2
    mu : float
        Expected upward drift per year, e.g. 0.08 = 8% per year
    sigma : float
        Volatility
    dt : float
        Length of each timestep in years, e.g. 1/12 = monthly
    n_steps : int
        Number of timesteps to simulate
    rho : int
        Correlation coefficient between the two paths
    
    Returns
    -------
    :obj:`~tensorflow.Tensor`
        2D array of paths, with shape `(n_steps + 1, n_paths)`, starting at `S0`
    """

    S0 = tf.reshape(S0, (1, 2))
    z = tf.random.normal((n_steps, 2), 0, dt ** 0.5)
    w1 = z[:, 0]
    w2 = rho * w1 + tf.sqrt(1 - rho * rho) * z[:, 1]
    w = tf.stack((w1, w2), axis=1)
    log_spots = tf.math.cumsum((mu - sigma * sigma / 2.0) * dt + sigma * w)
    spots = S0 * tf.math.exp(log_spots)
    return tf.concat((S0, spots), axis=0)
