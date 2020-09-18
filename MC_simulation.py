import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='darkgrid')
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

seed = 2
np.random.seed(seed)

n_steps = 100  # number of time steps between t=0 and t=texp years
n_paths = 10000  # number of MC paths to include in one step of the neural network training

S0 = 1  # initial spot price
mu = 0.0  # Same order of magnitude as `sqrtdt`: 0.1 is 10% per year
vol = 0.2
texp = 10  # time to option expiration in years
dt = texp / n_steps
sqrtdt = dt ** 0.5


def correlation(x, y):
    z = np.multiply(x, y)

    return (np.mean(z) - np.mean(x) * np.mean(y)) / (np.std(x) * np.std(y))


def generate_paths(n_paths=1, init_spot=1.0, n_steps=100, texp=1.0, vol=0.2, mu=0.0):
    """Generates spot curves using simple geometric Brownian motion.

    Parameters
    ----------
    n_paths : int
        The number of paths to generate
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

    return spot


def plot_paths(n_paths=1, init_spot=1.0, n_steps=100, texp=1.0, vol=0.2, mu=0.0):
    """ Plot paths of a Monte Carlo simulation """

    T_index = n_steps - 1
    paths = generate_paths(n_paths, init_spot, n_steps, texp, vol, mu)

    log.info('Average final spot %.2f', np.mean(paths[T_index, :]))

    plt.plot(paths)
    plt.title('Monte Carlo spot prices over time')
    plt.xlabel('Steps')
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


def generate_correlated_paths(n_paths, init_spot=(1.0, 1.0), n_steps=100, texp=10, vol=(0.2, 0.2), mu=(0.0, 0.0), rho=0.5):

    """Generates correlated spot curves using simple geometric Brownian motion."""

    init_spot1, init_spot2 = init_spot
    vol1, vol2 = vol
    mu1, mu2 = mu

    log_spot1 = np.zeros(n_paths)
    spot1 = np.zeros((n_steps, n_paths))

    log_spot2 = np.zeros(n_paths)
    spot2 = np.zeros((n_steps, n_paths))

    dt = texp / n_steps
    sqrtdt = dt ** 0.5

    for t in range(n_steps):

        rs = np.random.normal(0, sqrtdt, size=(n_paths, 2))

        w1 = rs[:, 0]
        w2 = rho * w1 + ((1 - rho * rho) ** 0.5) * rs[:, 1]

        log_spot1 += (mu1 - vol1 * vol1 / 2.0) * dt + vol1 * w1
        spot1[t, :] = init_spot1 * np.exp(log_spot1)

        log_spot2 += (mu2 - vol2 * vol2 / 2.0) * dt + vol2 * w2
        spot2[t, :] = init_spot2 * np.exp(log_spot2)

    return spot1, spot2


def plot_correlated_paths(n_paths=100, init_spot=(1.0, 1.0), n_steps=100, texp=1.0, vol=(0.2, 0.2), mu=(0.0, 0.0), rho=0.5):

    """Plot correlated paths of a Monte Carlo simulation.   """

    T_index = n_steps - 1
    paths1, paths2 = generate_correlated_paths(n_paths, init_spot, n_steps, texp, vol, mu, rho)

    log.info('Average final spot 1 %.2f', np.mean(paths1[T_index, :]))
    log.info('Average final spot 2 %.2f', np.mean(paths2[T_index, :]))
    log.info('Variance final spot 1 %.2f', np.var(paths1[T_index, :]))
    log.info('Variance final spot 2 %.2f', np.var(paths2[T_index, :]))
    log.info('Correlation between the two spots %.2f', correlation(np.log(paths1[T_index, :]), np.log(paths2[T_index, :])))

    plt.plot(paths1)
    plt.title('Monte Carlo spot prices over time (spot1)')
    plt.xlabel('Time')
    plt.ylabel('Spot')
    plt.gca().set_xlim([0, T_index])
    plt.show()

    plt.plot(paths2)
    plt.title('Monte Carlo spot prices over time (spot2)')
    plt.xlabel('Time')
    plt.ylabel('Spot')
    plt.gca().set_xlim([0, T_index])
    plt.show()


def plot_two_spots(n_paths=1, init_spot=(1.0, 1.0), n_steps=100, texp=1.0, vol=(0.2, 0.2), mu=(0.0, 0.0), rho=0.5):

    paths1, paths2 = generate_correlated_paths(n_paths, init_spot, n_steps, texp, vol, mu, rho)
    n = np.random.randint(0, n_paths)

    p1, p2 = paths1[:, n], paths2[:, n]

    plt.plot(p1)
    plt.plot(p2)
    plt.xlabel('Steps')
    plt.ylabel('Spot')
