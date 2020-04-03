"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Mark Higgins, Ben Pryke
Summary: Deep hedging example implementation for pricing a vanilla option under BS
Description: |
    Try out the deep hedging example from paper https://arxiv.org/abs/1802.03042.
    
    That paper lays out a general scheme for both pricing a derivative portfolio and
    getting the fair upfront premium. It then applies it to an example of pricing a 
    vanilla option under the Heston model.
    
    This example applies to a simpler case: pricing a vanilla option under the Black-Scholes
    model. In this case there is just one hedge instrument: the asset itself (for a delta
    hedge). We also leave off transaction costs, so this should reproduce standard BS
    pricing in the limit of continuous hedging.
    
    Under these standard risk neutral assumptions, the thing that minimises expected
    shortfall is continuous hedging using the Black-Scholes delta. This example asks the
    question: in the risk neutral limit, can a neural network independently learn to
    approximate the Black-Scholes delta when trained to minimise expected shortfall?
    The example is useful because it proves out the case in which we know what the answer
    should be.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import tensorflow as tf

from models.variable_annuity.model import VariableAnnuity, simulate, set_seed, log_training_progress
from plotting import ResultTypes, plot_heatmap, plot_deltas, plot_loss, plot_pnls

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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


def run_once(do_train=True, show_loss_plot=False, show_delta_plot=True, show_pnl_plot=True, **hparams):
    """
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
    # model.restore()
    
    if do_train:
        losses = model.train(post_batch_callback=log_training_progress)
        
        if show_loss_plot:
            plot_loss(losses)
    
    if show_delta_plot:
        plot_deltas(model)
    
    if show_pnl_plot:
        log.info('Testing on %d paths', model.n_test_paths)
        pnls = simulate(model)
        plot_pnls(pnls, types=(ResultTypes.UNHEDGED, ResultTypes.BLACK_SCHOLES, ResultTypes.DEEP_HEDGING))


if __name__ == '__main__':
    set_seed(2)
    run_once(learning_rate=5e-3, n_batches=100, mu=0.0, vol=0.2, n_test_paths=1_000, S0=1.)
    # run_once(learning_rate=1e-3, n_batches=5000, mu=0.0, vol=0.2, n_test_paths=10_000, n_layers=2, n_hidden=25)
    # search_vol_vs_mu()
    # plot_heatmap(
    #     model=VariableAnnuity,
    #     title='Deep Hedging error vs Black-Scholes',
    #     xparam='learning_rate',
    #     xlabel='Learning rate',
    #     xvals=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    #     yparam='n_batches',
    #     ylabel='Training batches',
    #     yvals=[100, 500, 1000, 5000, 10000],
    # )

    # plot_heatmap(
    #     model=VariableAnnuity,
    #     title='Deep Hedging error vs Black-Scholes',
    #     xparam='n_layers',
    #     xlabel='Hidden layers',
    #     xvals=[1, 2, 3, 4, 5],
    #     yparam='n_hidden',
    #     ylabel='Hidden units per layer',
    #     yvals=[5, 25, 50, 100, 200],
    #     learning_rate=1e-2,
    #     n_batches=5000,
    # )

    # plot_heatmap(
    #     model=VariableAnnuity,
    #     title='Deep Hedging error vs Black-Scholes',
    #     xparam='beta_1',
    #     xlabel='Adam Beta 1',
    #     xvals=[0.6, 0.7, 0.8, 0.9, 0.99],
    #     yparam='learning_rate',
    #     ylabel='Learning rate',
    #     yvals=[1e-2, 1e-3],
    #     n_batches=5000,
    # )
