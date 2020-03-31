"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Mark Higgins, Ben Pryke
Description: |
    Plotting for the Variable Annuity model.
"""

from enum import Enum
import logging
import time

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

import lib.black_scholes as bs
from lib.utils import get_duration_desc
from va.model import Model, compute_analytical_bs_delta, test, train

sns.set(style='darkgrid', palette='deep')
plt.rcParams['figure.figsize'] = (8, 6)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ResultTypes(Enum):
    """Result identifier with `label` and `colour` fields for each type.
    
    For consistency between different plots of data corresponding to the same
    unerlying models.
    """
    
    def __new__(cls, label, colour_index):
        palette = sns.color_palette()
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.colour = palette[colour_index]
        return obj
    
    UNHEDGED = ('Unhedged', 2)
    BLACK_SCHOLES = ('Black-Scholes', 1)
    DEEP_HEDGING = ('Deep Hedging', 0)


def calc_thist(data, n_bins=30):
    n_steps = len(data)
    x = np.array([np.arange(n_bins) for _ in range(n_steps)]).flatten()
    y = np.array([np.full(n_bins, i) for i in range(n_steps)]).flatten()
    z = np.zeros(n_bins * n_steps)
    dx = dy = np.zeros(n_bins * n_steps) + 0.5
    flat_data = [i for j in data for i in j]
    r = (min(flat_data), max(flat_data))
    dz = np.array([np.histogram(d, n_bins, r)[0] for d in data])[::-1].flatten()
    return x, y, z, dx, dy, dz, r


def plot_thist(data, n_bins=30):
    from mpl_toolkits.mplot3d import Axes3D
    n_ticks = 7
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    *bar_data, r = calc_thist(data)
    ax.bar3d(*bar_data, linewidth=0.1)
    plt.title('Delta distribution over time')
    ax.set_xlabel('Delta')
    ax.set_xticks(np.arange(0, n_bins + 1, n_bins // (n_ticks - 1)))
    ax.set_xticklabels([f'{r[0] + i * abs(r[0] - r[1]) / (n_ticks - 1):.2f}' for i in range(n_ticks)])
    ax.set_ylabel('Simulation Month')
    ax.set_yticks(np.arange(0, len(data) + 1, len(data) // (n_ticks - 1)))
    ax.set_yticklabels([f'{(n_ticks - i) * 10:d}' for i in range(1, n_ticks + 1)])
    ax.set_zlabel('Frequency')
    plt.show()


def plot_loss(losses, window1=50, window2=500):
    smoothed1 = np.convolve(losses, np.ones((window1,)) / window1, mode='valid')
    smoothed2 = np.convolve(losses, np.ones((window2,)) / window2, mode='valid')
    plt.plot(losses)
    plt.plot(smoothed1)
    plt.plot(smoothed2)
    plt.title('Loss over time')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(labels=['Loss', f'Mean of {window1}', f'Mean of {window2}'])
    plt.tight_layout()
    plt.show()


def plot_deltas(model):
    """Plot out delta vs spot for a range of calendar times
    
    Calculated against the known closed-form BS delta.
    """
    
    f, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    f.suptitle('Delta hedge vs spot vs time to maturity')
    axes = axes.flatten()
    spot_fact = np.exp(3 * model.vol * model.texp ** 0.5)
    ts = [0., model.texp * 0.25, model.texp * 0.5, model.texp * 0.95]
    n_spots = 20
    
    for t, ax in zip(ts, axes):
        # Compute neural network delta
        test_spot = np.linspace(model.S0 / spot_fact, model.S0 * spot_fact, n_spots).astype(np.float32)
        test_input = np.transpose(np.array([test_spot, [t] * n_spots], dtype=np.float32))
        
        # Compute neural network delta
        test_delta = model.compute_hedge_delta(test_input)[:, 0].numpy()
        test_delta = np.minimum(test_delta, 0) # pylint: disable=assignment-from-no-return
        test_delta *= (1 - np.exp(-model.lam * (model.texp - t))) * model.principal
        log.info('Delta: mean = % .5f, std = % .5f', test_delta.mean(), test_delta.std())
        
        # Compute Black Scholes delta
        # The hedge will have the opposite sign as the option we are hedging,
        # ie the hedge of a long call is a short call, so we flip psi.
        account = model.principal * test_spot / model.S0 * np.exp(-model.fee * t)
        est_deltas = compute_analytical_bs_delta(model.texp, t, model.lam, model.vol, model.fee, model.gmdb, account, test_spot)
        
        # Add a subsplot
        ax.set_title('Calendar time {:.2f} years'.format(t))
        bs_plot, = ax.plot(test_spot, est_deltas, color=ResultTypes.BLACK_SCHOLES.colour)
        nn_plot, = ax.plot(test_spot, test_delta, color=ResultTypes.DEEP_HEDGING.colour)
    
    ax.legend([bs_plot, nn_plot], [ResultTypes.BLACK_SCHOLES.label, ResultTypes.DEEP_HEDGING.label])
    f.text(0.5, 0.04, 'Spot', ha='center')
    f.text(0.04, 0.5, 'Delta', ha='center', rotation='vertical')
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.show()


def plot_pnls(pnls, types, *, trim_tails=0):
    """Plot histogram comparing pnls
    
    pnls : list of :obj:`numpy.array`
        Pnls to plot
    types : list of `ResultTypes`
        Type for the pnl data at each corresponding index in `pnls`
    trim_tails : int
        Percentile to trim from each tail when plotting
    """
    
    hist_range = (np.percentile(pnls, trim_tails), np.percentile(pnls, 100 - trim_tails))
    
    for pnl, rtype in zip(pnls, types):
        face_color = matplotlib.colors.to_rgba(rtype.colour, 0.7)
        plt.hist(pnl, range=hist_range, bins=200, facecolor=face_color, edgecolor=(1, 1, 1, 0.01), linewidth=0.5)
    
    plt.title('Post-simulation PNL histograms')
    plt.xlabel('PNL')
    plt.ylabel('Frequency')
    plt.legend([rtype.label for rtype in types])
    plt.tight_layout()
    plt.show()


def plot_heatmap(data, title, xlabel, xticklabels, ylabel, yticklabels):
    # yticklabels = np.flip(yticklabels, axis=0)
    # data = np.flip(data, axis=0)
    sns.heatmap(data=data[::-1], annot=True, fmt=".4f", xticklabels=xticklabels, yticklabels=yticklabels[::-1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def grid_search(title, xparam, xlabel, xvals, yparam, ylabel, yvals, *, repeats=3, **kwargs):
    """Run the model"""
    
    t0 = time.time()
    log.info('Hedging a variable annuity')
    
    errors = np.zeros((len(yvals), len(xvals)))
    
    for i, y in enumerate(yvals):
        for j, x in enumerate(xvals):
            log.info('Training with (y=%f, x=%f) over %d repeats', y, x, repeats)
            hparams = dict({xparam: x, yparam: y}, **kwargs)
            
            for _ in range(repeats):
                model = Model(**hparams)
                train(model)
                errors[i, j] += test(model)
            
            errors[i, j] /= repeats
            log.info('Error for (y=%f, x=%f): %.5f', y, x, errors[i, j])
    
    log.info('Results')
    log.info(np.array2string(errors, separator=','))
    log.info('Total running time: %s', get_duration_desc(t0))
    
    plot_heatmap(errors, title, xlabel, xvals, ylabel, yvals)


def search_vol_vs_mu():
    # mus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    # vols = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    vols = [0.05, 0.1, 0.15, 0.2, 0.25]
    mus = [0.0, 0.05, 0.1, 0.15]
    grid_search(
        title='Deep Hedging error vs Black-Scholes',
        xparam='vol',
        xlabel='Vol',
        xvals=vols,
        yparam='mu',
        ylabel='Expected annual drift',
        yvals=mus,
    )
