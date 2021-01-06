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
        2D tensor of paths, with shape `(n_steps + 1, n_paths)`, each starting at `S0`
    """

    S0 = tf.fill((1, n_paths), S0)
    w = tf.random.normal((n_steps, n_paths), 0, dt ** 0.5)
    log_spots = tf.math.cumsum((mu - sigma * sigma / 2.0) * dt + sigma * w)
    spots = S0 * tf.math.exp(log_spots)
    return tf.concat((S0, spots), axis=0)


@tf.function
def gbm2(S0, mu, sigma, dt, rho, n_steps, n_paths=1):
    """Simulates correlated geometric Brownian motion (GBM) for 2 paths `n_paths` times.
    
    Parameters
    ----------
    S0 : float, list, tuple, :obj:`~numpy.array`, or :obj:`~tensorflow.Tensor`
        Initial values, of length 2.
        Passing an iterable of length 2 will give each pair separate initial values.
    mu : float, list, tuple, :obj:`~numpy.array`, or :obj:`~tensorflow.Tensor`
        Expected upward drift per year, e.g. 0.08 = 8% per year.
        Passing an iterable of length 2 will give each pair separate drifts.
    sigma : float, list, tuple, :obj:`~numpy.array`, or :obj:`~tensorflow.Tensor`
        Volatility.
        Passing an iterable of length 2 will give each pair separate vols.
    dt : float
        Length of each timestep in years, e.g. 1/12 = monthly
    n_steps : int
        Number of timesteps to simulate
    rho : int
        Correlation coefficient between the two paths
    
    Returns
    -------
    :obj:`~tensorflow.Tensor`
        3D tensor of paths, with shape `(n_steps + 1, n_paths, 2)`, starting at `S0`
    """

    S0 = tf.broadcast_to(S0, (1, n_paths, 2))
    mu = tf.broadcast_to(mu, (n_steps, n_paths, 2))
    sigma = tf.broadcast_to(sigma, (n_steps, n_paths, 2))

    z = tf.random.normal((n_steps, n_paths, 2), 0, dt ** 0.5)
    w1 = z[:, :, 0]
    w2 = rho * w1 + tf.sqrt(1.0 - rho * rho) * z[:, :, 1]
    w = tf.stack((w1, w2), axis=2)
    log_spots = tf.math.cumsum((mu - sigma * sigma / 2.0) * dt + sigma * w)
    spots = S0 * tf.math.exp(log_spots)
    return tf.concat((S0, spots), axis=0)
