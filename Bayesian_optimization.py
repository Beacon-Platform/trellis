"""
Script to run bayesian optimization for European and heat rate options
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import tensorflow as tf
import xlsxwriter

from models import EuropeanOption, HeatrateOption
from plotting import ResultTypes, plot_heatmap, plot_deltas, plot_loss, plot_pnls
from utils import get_progressive_min

from bayes_opt import BayesianOptimization


def get_callbacks(model):
    return [
        tf.keras.callbacks.ModelCheckpoint(model.checkpoint_prefix, monitor='val_loss', save_best_only=True),
    ]


def loss_function_default(**hparams):
    """ Loss function of European options  """

    int_params = ('n_layers', 'n_hidden', 'batch_size', 'epoch_size', 'n_epochs', 'n_val_paths', 'n_test_paths')

    for key in hparams:
        if key in int_params:
            hparams[key] = int(hparams[key])

    model = EuropeanOption(**hparams)

    history = model.train(callbacks=get_callbacks(model), verbose=0)
    ES = get_progressive_min(model.history.history['val_loss'])[-1]

    return -ES


def bayes_optimization(pbounds, file_name):

    """Run a Bayesian optimization for European options.

    Parameters
    ----------

    pbounds: dictionary
        Bounded region of parameter space
    file_name: string ('<name>.xlsx')
        Name of excel file

    """

    int_params = ('n_layers', 'n_hidden', 'batch_size', 'epoch_size', 'n_epochs', 'n_val_paths', 'n_test_paths')

    optimizer = BayesianOptimization(
        f=loss_function_default,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum
        # is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=15)
    print(optimizer.max)

    opt_params = {}

    for i in range(len(optimizer.space.keys)):
        opt_params[optimizer.space.keys[i]] = optimizer.space.params[:, i]

    loss_target = optimizer.space.target

    # Save Bayesian optimization results in a excel file
    workbook = xlsxwriter.Workbook(file_name)
    sheet = workbook.add_worksheet()

    sheet.write(0, 0, 'loss')
    for i in range(len(loss_target)):
        sheet.write(i + 1, 0, loss_target[i])

    t = 1
    for k in opt_params:
        sheet.write(0, t, k)
        if k in int_params:
            for j in range(len(opt_params[k])):
                sheet.write(j + 1, t, int(opt_params[k][j]))
            t = t + 1
        else:
            for j in range(len(opt_params[k])):
                sheet.write(j + 1, t, opt_params[k][j])
            t = t + 1

    workbook.close()


def loss_function_heatrate(**hparams):

    """ Loss function of heat rate options  """

    int_params = ('n_layers', 'n_hidden', 'batch_size', 'epoch_size', 'n_epochs', 'n_val_paths', 'n_test_paths')

    for key in hparams:
        if key in int_params:
            hparams[key] = int(hparams[key])

    model = HeatrateOption(**hparams)

    history = model.train(callbacks=get_callbacks(model), verbose=0)
    ES = get_progressive_min(model.history.history['val_loss'])[-1]

    return -ES


def bayes_optimization_heatrate(pbounds, file_name):

    """Run a Bayesian optimization for heat rate options.

    Parameters
    ----------

    pbounds: dictionary
        Bounded region of parameter space
    file_name: string ('<name>.xlsx')
        Name of excel file

    """

    int_params = ('n_layers', 'n_hidden', 'batch_size', 'epoch_size', 'n_epochs', 'n_val_paths', 'n_test_paths')

    optimizer = BayesianOptimization(
        f=loss_function_heatrate,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum
        # is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=15)
    print(optimizer.max)

    opt_params = {}

    for i in range(len(optimizer.space.keys)):
        opt_params[optimizer.space.keys[i]] = optimizer.space.params[:, i]

    loss_target = optimizer.space.target

    # Save Bayesian optimization results in a excel file
    workbook = xlsxwriter.Workbook(file_name)
    sheet = workbook.add_worksheet()

    sheet.write(0, 0, 'loss')
    for i in range(len(loss_target)):
        sheet.write(i + 1, 0, loss_target[i])

    t = 1
    for k in opt_params:
        sheet.write(0, t, k)
        if k in int_params:
            for j in range(len(opt_params[k])):
                sheet.write(j + 1, t, int(opt_params[k][j]))
            t = t + 1
        else:
            for j in range(len(opt_params[k])):
                sheet.write(j + 1, t, opt_params[k][j])
            t = t + 1

    workbook.close()
