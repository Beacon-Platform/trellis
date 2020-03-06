"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Ben Pryke
Description: Plots Monte Carlo simulations of spot price over time.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='darkgrid')
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

seed = 2
np.random.seed(seed)

n_steps = 120 # number of time steps between t=0 and t=texp years
n_paths = 10000 # number of MC paths to include in one step of the neural network training

S0 = 1 # initial spot price
mu = 0.0 # Same order of magnitude as `sqrtdt`: 0.1 is 10% per year
vol = 0.2
texp = 10 # time to option expiration in years
dt = texp / n_steps
sqrtdt = dt ** 0.5


def generate_paths(n_paths=1, init_spot=1.0, n_steps=100, texp=1.0, vol=0.2, mu=0.0):
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
    
    return spot


def plot_paths(n_paths=1, init_spot=1.0, n_steps=100, texp=1.0, vol=0.2, mu=0.0):
    T_index = n_steps - 1
    paths = generate_paths(n_paths, init_spot, n_steps, texp, vol, mu)
    
    log.info('Average final spot %.2f', np.mean(paths[T_index, :]))
    
    plt.plot(paths)
    plt.title('Monte Carlo spot prices over time')
    plt.xlabel('Time')
    plt.ylabel('Spot')
    plt.gca().set_xlim([0, T_index])
    plt.show()


def plot_spot_hist(t, n_paths=1, init_spot=1.0, n_steps=100, texp=1.0, vol=0.2, mu=0.0):
    T_index = 0 if t == 0 else int(t / texp * n_steps) - 1
    paths = generate_paths(n_paths, init_spot, n_steps, texp, vol, mu)
    
    log.info('Average final spot %.2f', np.mean(paths[T_index, :]))
    
    plt.hist(paths[T_index, :], bins=50)
    plt.title('Histogram of spot prices at time {:.2f} year(s)'.format(t))
    plt.xlabel('Spot')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_paths(100, S0, n_steps, texp, vol, mu)
    plot_spot_hist(texp, 10000, S0, n_steps, texp, vol, mu)
