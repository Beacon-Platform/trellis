# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Deep hedging example entry-point for pricing a vanilla option under BS."""

from utils import disable_gpu
disable_gpu() # Call first

import logging

import numpy as np
import tensorflow as tf

import models.european_option.analytics as analytics
from models import EuropeanOption
from models.utils import set_seed, estimate_expected_shortfalls
from plotting import ResultTypes, plot_heatmap, plot_deltas, plot_loss, plot_pnls
from utils import get_progressive_min

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_callbacks(model):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model.checkpoint_prefix, monitor='val_loss', save_best_only=True),
    ]


def run_once(do_train=True, show_loss_plot=True, show_delta_plot=True, show_pnl_plot=True, **hparams):
    """Trains and tests a model, and displays some plots.
    
    Parameters
    ----------
    do_train : bool
        Actually train the model
    show_loss_plot : bool
        Pop plot of training loss
    show_delta_plot : bool
        Pop up plot of delta vs spot
    show_pnl_plot : bool
        Run MC sim to compute PnL
    """
    
    model = EuropeanOption(**hparams)
    
    if do_train:
        history = model.train(callbacks=get_callbacks(model))
        
        if show_loss_plot:
            plot_loss(get_progressive_min(history.history['val_loss']))
    
    model.restore()
    
    if show_delta_plot:
        def compute_nn_delta(model, t, spot):
            nn_input = np.transpose(np.array([spot, [t] * len(spot)], dtype=np.float32))
            return model.compute_hedge_delta(nn_input)[:, 0].numpy()
        
        def compute_bs_delta(model, t, spot):
            # The hedge will have the opposite sign as the option we are hedging,
            # ie the hedge of a long call is a short call, so we flip psi.
            return -model.psi * analytics.calc_opt_delta(model.is_call, spot, model.K, model.texp - t, model.vol, 0, 0)
        
        plot_deltas(model, compute_nn_delta, compute_bs_delta)
    
    if show_pnl_plot:
        log.info('Testing on %d paths', model.n_test_paths)
        pnls = model.simulate(n_paths=model.n_test_paths)
        estimate_expected_shortfalls(*pnls, pctile=model.pctile)
        plot_pnls(pnls, types=(ResultTypes.UNHEDGED, ResultTypes.BLACK_SCHOLES, ResultTypes.DEEP_HEDGING))


if __name__ == '__main__':
    set_seed(2)
    run_once(n_epochs=100, learning_rate=5e-3, mu=0.1, vol=0.2)
    
    # plot_heatmap(
    #     model=EuropeanOption,
    #     title='Deep Hedging error vs Black-Scholes',
    #     xparam='b_std',
    #     xlabel='Initial bias std',
    #     xvals=[0., 0.05, 0.1],
    #     yparam='learning_rate',
    #     ylabel='Learning rate',
    #     yvals=[1e-3, 5e-3, 1e-4],
    #     get_callbacks=get_callbacks,
    # )
