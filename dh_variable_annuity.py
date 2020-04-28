# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Deep hedging example entry-point for pricing a variable annuity under BS."""

from utils import disable_gpu
disable_gpu() # Call first

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import tensorflow as tf

from models.utils import set_seed, estimate_expected_shortfalls
import models.variable_annuity.analytics as analytics
from models.variable_annuity.model import VariableAnnuity
from plotting import ResultTypes, plot_heatmap, plot_deltas, plot_loss, plot_pnls
from utils import get_progressive_min

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_callbacks(model):
    return [
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model.checkpoint_prefix, monitor='val_loss', save_best_only=True),
    ]


def search_vol_vs_mu():
    # mus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    # vols = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    vols = [0.05, 0.1, 0.15, 0.2, 0.25]
    mus = [0.0, 0.05, 0.1, 0.15]
    plot_heatmap(
        model=VariableAnnuity,
        title='Deep Hedging error vs Black-Scholes',
        xparam='vol',
        xlabel='Vol',
        xvals=vols,
        yparam='mu',
        ylabel='Expected annual drift',
        yvals=mus,
    )


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
    
    model = VariableAnnuity(**hparams)
    
    if do_train:
        history = model.train(callbacks=get_callbacks(model))
        
        if show_loss_plot:
            plot_loss(get_progressive_min(history.history['val_loss']))
    
    model.restore()
    
    if show_delta_plot:
        def compute_nn_delta(model, t, spot):
            nn_input = np.transpose(np.array([spot, [t] * len(spot)], dtype=np.float32))
            delta = model.compute_hedge_delta(nn_input)[:, 0].numpy()
            delta = np.minimum(delta, 0) # pylint: disable=assignment-from-no-return
            delta *= (1 - np.exp(-model.lam * (model.texp - t))) * model.principal
            return delta
        
        def compute_bs_delta(model, t, spot):
            account = model.principal * spot / model.S0 * np.exp(-model.fee * t)
            return analytics.compute_delta(model.texp, t, model.lam, model.vol, model.fee, model.gmdb, account, spot)
        
        plot_deltas(model, compute_nn_delta, compute_bs_delta)
    
    if show_pnl_plot:
        log.info('Testing on %d paths', model.n_test_paths)
        pnls = model.simulate(n_paths=model.n_test_paths)
        estimate_expected_shortfalls(*pnls, pctile=model.pctile)
        plot_pnls(pnls, types=(ResultTypes.UNHEDGED, ResultTypes.BLACK_SCHOLES, ResultTypes.DEEP_HEDGING))


if __name__ == '__main__':
    set_seed(2)
    run_once(n_epochs=2, learning_rate=5e-3, mu=0.08, vol=0.2)
    # search_vol_vs_mu()
