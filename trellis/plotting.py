# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Plotting library for models."""

from enum import Enum
import logging
import time

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from trellis.utils import get_duration_desc

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


def plot_loss(losses, *, smoothing_windows=(5, 25), min_points=10):
    """Plot loss against number of epochs, maybe adding smoothed averages"""

    curves = [losses]
    labels = ['Loss']

    for window in smoothing_windows:
        if len(curves) > window * min_points:
            smoothed = np.convolve(losses, np.ones((window,)) / window, mode='valid')
            curves.append(smoothed)
            labels.append(f'Mean of {window}')

    for curve in curves:
        plt.plot(range(1, len(losses) + 1), curve)

    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(labels=labels)
    plt.tight_layout()
    plt.gca().set_xlim([1, len(losses)])
    plt.show()


def plot_deltas(model, compute_nn_delta, compute_bs_delta, *, verbose=1):
    """Plot out delta vs spot for a range of calendar times
    
    Calculated against the known closed-form BS delta.
    """

    f, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    f.suptitle('Delta hedge vs spot vs time to maturity')
    axes = axes.flatten()
    spot_fact = np.exp(3 * model.vol * model.texp ** 0.5)
    ts = [0.0, model.texp * 0.25, model.texp * 0.5, model.texp * 0.95]
    n_spots = 1000

    for t, ax in zip(ts, axes):
        spot_min = model.S0 / spot_fact
        spot_max = model.S0 * spot_fact
        test_spot = np.linspace(spot_min, spot_max, n_spots).astype(np.float32)
        test_delta = compute_nn_delta(model, t, test_spot)
        est_delta = compute_bs_delta(model, t, test_spot)

        if verbose != 0:
            log.info('Delta: mean = % .5f, std = % .5f', test_delta.mean(), test_delta.std())

        # Add a subsplot
        ax.set_title('Calendar time {:.2f} years'.format(t))
        ax.set_xlim([spot_min, spot_max])
        (bs_plot,) = ax.plot(test_spot, est_delta, color=ResultTypes.BLACK_SCHOLES.colour)
        (nn_plot,) = ax.plot(test_spot, test_delta, color=ResultTypes.DEEP_HEDGING.colour)

    ax.legend([bs_plot, nn_plot], [ResultTypes.BLACK_SCHOLES.label, ResultTypes.DEEP_HEDGING.label])
    f.text(0.5, 0.04, 'Spot', ha='center')
    f.text(0.04, 0.5, 'Delta', ha='center', rotation='vertical')
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.show()


def plot_deltas_heatrate(
    filename, model, compute_nn_delta, compute_bs_delta, delta_type, spot_type, *, verbose=1,
):
    """Plot out four plots of delta (power/gas) vs spot (power/gas) for a range of calendar times"""

    f, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    f.suptitle('Delta hedge ' + delta_type + ' vs spot ' + spot_type + ' vs time to maturity')
    axes = axes.flatten()
    ts = [0.0, model.texp * 0.25, model.texp * 0.5, model.texp * 0.95]
    n_spots = 1000

    for t, ax in zip(ts, axes):
        if spot_type == 'power':
            spot_min_power = model.SP0 - 10
            spot_max_power = model.SP0 + 10
            spot_min_gas = spot_max_gas = model.SG0
        else:
            spot_min_gas = model.SG0 - 10
            spot_max_gas = model.SG0 + 10
            spot_min_power = spot_max_power = model.SP0

        test_spot_power = np.linspace(spot_min_power, spot_max_power, n_spots).astype(np.float32)
        test_spot_gas = np.linspace(spot_min_gas, spot_max_gas, n_spots).astype(np.float32)

        test_delta = compute_nn_delta(model, t, test_spot_power, test_spot_gas, delta_type)
        est_delta = compute_bs_delta(model, t, test_spot_power, test_spot_gas, delta_type)

        if verbose != 0:
            log.info('Delta: mean = % .5f, std = % .5f', test_delta.mean(), test_delta.std())

        # Add a subsplot
        ax.set_title('Calendar time {:.2f} years'.format(t))

        if spot_type == 'power':
            ax.set_xlim([spot_min_power, spot_max_power])
            (bs_plot,) = ax.plot(test_spot_power, est_delta, color=ResultTypes.BLACK_SCHOLES.colour)
            (nn_plot,) = ax.plot(test_spot_power, test_delta, color=ResultTypes.DEEP_HEDGING.colour)
        else:
            ax.set_xlim([spot_min_gas, spot_max_gas])
            (bs_plot,) = ax.plot(test_spot_gas, est_delta, color=ResultTypes.BLACK_SCHOLES.colour)
            (nn_plot,) = ax.plot(test_spot_gas, test_delta, color=ResultTypes.DEEP_HEDGING.colour)

    ax.legend([bs_plot, nn_plot], [ResultTypes.BLACK_SCHOLES.label, ResultTypes.DEEP_HEDGING.label])
    f.text(0.5, 0.04, 'Spot ' + spot_type, ha='center')
    f.text(0.04, 0.5, 'Delta ' + delta_type, ha='center', rotation='vertical')
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.savefig(filename)


def plot_pnls(pnls, types, *, trim_tails=0):
    """Plot histogram comparing pnls
    
    pnls : list of :obj:`numpy.array`
        Pnls to plot
    types : list of `ResultTypes`
        Type for the pnl data at each corresponding index in `pnls`
    trim_tails : int
        Percentile to trim from each tail when plotting
    """

    _ = plt.figure()
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


def compute_heatmap(model, title, xparam, xvals, yparam, yvals, *, repeats=3, get_callbacks=None, **kwargs):
    """Run the model"""

    t0 = time.time()
    log.info('Hedging a variable annuity')

    errors = np.zeros((len(yvals), len(xvals)))

    for i, y in enumerate(yvals):
        for j, x in enumerate(xvals):
            log.info('Training with (y=%f, x=%f) over %d repeats', y, x, repeats)
            hparams = dict({xparam: x, yparam: y}, **kwargs)

            for _ in range(repeats):
                mdl = model(**hparams)
                callbacks = get_callbacks(mdl) if get_callbacks is not None else None
                mdl.train(callbacks=callbacks)
                errors[i, j] += mdl.test()

            errors[i, j] /= repeats
            log.info('Test error for (y=%f, x=%f): %.5f', y, x, errors[i, j])

    log.info('Heatmap:')
    log.info(np.array2string(errors, separator=','))
    log.info('Total running time: %s', get_duration_desc(t0))

    return errors


def plot_heatmap(model, title, xparam, xlabel, xvals, yparam, ylabel, yvals, *, repeats=3, **kwargs):
    errors = compute_heatmap(model, title, xparam, xvals, yparam, yvals, repeats=repeats, **kwargs)
    sns.heatmap(data=errors[::-1], annot=True, fmt=".4f", xticklabels=xvals, yticklabels=yvals[::-1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_paths(paths):
    """Plot many paths to visualise Monte Carlo simulation over time."""

    n_paths = len(paths) - 1
    plt.plot(paths)
    plt.title('Monte Carlo simulation of spot prices over time')
    plt.xlabel('Time')
    plt.ylabel('Spot')
    plt.gca().set_xlim([0, n_paths])
    plt.show()


def plot_spot_hist(paths, time_index):
    """Plot histogram of spot prices at a given index in the simulation."""

    plt.hist(paths[time_index, :], bins=50)
    plt.title('Histogram of spot prices at simulation step {}'.format(time_index))
    plt.xlabel('Spot')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
