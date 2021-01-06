# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins, Amine Benchrifa

"""Deep hedging example entry-point for pricing a heat rate option under BS."""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import trellis.models.heatrate_option.analytics as analytics
from trellis.models.heatrate_option.model import HeatrateOption
from trellis.models.utils import set_seed, estimate_expected_shortfalls
from trellis.plotting import ResultTypes, plot_heatmap, plot_deltas_heatrate, plot_loss, plot_pnls
from trellis.utils import get_progressive_min

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_callbacks(model):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model.checkpoint_prefix, monitor='val_loss', save_best_only=True),
    ]


def run_once(plot_path='./plots/', do_train=True, show_loss_plot=True, show_delta_plot=True, show_pnl_plot=True, **hparams):
    """Trains and tests a model, and displays some plots.

    Parameters
    ----------
    plot_path : str
        path to write the plots
    do_train : bool
        Actually train the model
    show_loss_plot : bool
        Pop plot of training loss
    show_delta_plot : bool
        Pop up plot of delta vs spot
    show_pnl_plot : bool
        Run MC sim to compute PnL
    """

    model = HeatrateOption(**hparams)

    if do_train:
        history = model.train(callbacks=get_callbacks(model))

        if show_loss_plot:
            plot_loss(get_progressive_min(history.history['val_loss']))

    model.restore()

    if show_delta_plot:

        def compute_nn_delta(model, t, spot_power, spot_gas, delta_type):
            nn_input = np.transpose(np.array([spot_power, spot_gas, [t] * len(spot_power)], dtype=np.float32))

            output_index = 0 if delta_type == 'power' else 1

            return model.compute_hedge_delta(nn_input)[:, output_index].numpy()

        def compute_bs_delta(model, t, spot_power, spot_gas, delta_type):
            # The hedge will have the opposite sign as the option we are hedging,
            # ie the hedge of a long call is a short call, so we flip psi.
            deltas = analytics.calc_opt_delta(
                model.is_call, spot_power, spot_gas, model.K, model.H, model.texp - t, model.vol_P, model.vol_G, model.mu_P, model.rho
            )

            if delta_type == 'power':
                return -model.psi * deltas[0]
            else:
                return -model.psi * deltas[1]

        Path(plot_path).mkdir(parents=True, exist_ok=True)
        plot_path += model.model_id + '-' if model.model_id else ''
        plot_path += 'delta-plot-'

        plot_deltas_heatrate(plot_path + 'power-power', model, compute_nn_delta, compute_bs_delta, 'power', 'power')
        plot_deltas_heatrate(plot_path + 'power-gas', model, compute_nn_delta, compute_bs_delta, 'power', 'gas')
        plot_deltas_heatrate(plot_path + 'gas-power', model, compute_nn_delta, compute_bs_delta, 'gas', 'power')
        plot_deltas_heatrate(plot_path + 'gas-gas', model, compute_nn_delta, compute_bs_delta, 'gas', 'gas')

    if show_pnl_plot:
        log.info('Testing on %d paths', model.n_test_paths)
        pnls = model.simulate(n_paths=model.n_test_paths)
        estimate_expected_shortfalls(*pnls, pctile=model.pctile)
        plot_pnls(pnls, types=(ResultTypes.UNHEDGED, ResultTypes.BLACK_SCHOLES, ResultTypes.DEEP_HEDGING))


if __name__ == '__main__':
    set_seed(2)
    run_once()
