# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Base classes for models."""

from hashlib import md5
import logging
import os
import time

from tensorflow.keras.callbacks import CallbackList
import tensorflow as tf

from trellis.utils import calc_expected_shortfall, get_duration_desc

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ROOT_CHECKPOINT_DIR = './checkpoints/'


class HyperparamsBase:
    root_checkpoint_dir = ROOT_CHECKPOINT_DIR
    model_id = None  # Unique id that, if defined, forms part of the checkpoint directory name

    learning_rate = 5e-3
    batch_size = 100  # Number of MC paths per batch
    epoch_size = 100  # Number of batches per epoch
    n_epochs = 100  # Number of epochs to train for
    n_val_paths = 50_000  # Number of paths to validate against
    n_test_paths = 100_000  # Number of paths to test against

    def __init__(self, **kwargs):
        # Set hyperparameters
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError('Invalid hyperparameter "%s"', k)

    @property
    def critical_fields(self):
        """Tuple of parameters that uniquely define the model."""
        raise NotImplementedError()

    @property
    def checkpoint_directory(self):
        """Directory in which to save checkpoint files."""
        base = '{}{}_'.format(self.root_checkpoint_dir, self.__class__.__name__)

        if self.model_id is not None:
            return base + self.model_id

        return base + md5(str(hash(self.critical_fields)).encode('utf-8')).hexdigest()

    @property
    def checkpoint_prefix(self):
        """Filepath prefix to be used when saving/loading checkpoints.
        
        Includes everything except for the file extension.
        """
        return os.path.join(self.checkpoint_directory, 'checkpoint')


class Model(tf.keras.Sequential):
    def __init__(self):
        tf.keras.Sequential.__init__(self)

    def train(self, optimizer=None, callbacks=None, *, verbose=1):
        """Train the network by running MC simulations, validating every `self.epoch_size` epochs.
        
        Note: these epochs are not true epochs in the sense of full passes over the dataset, as
        our dataset is infinite and we continuously generate new data. However, we need to
        evaluate our progress as training continues, hence the introduction of `epoch_size`.
        """

        n_paths = self.batch_size * self.epoch_size * self.n_epochs

        if verbose != 0:
            log.info('Training on %d paths over %d epochs', n_paths, self.n_epochs)

        if not isinstance(callbacks, CallbackList):
            callbacks = list(callbacks or [])
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                do_validation=True,
                verbose=verbose,
                epochs=self.n_epochs,
                steps=self.epoch_size,
            )

        # Use the Adam optimizer y default, which is gradient descent which also evolves
        # the learning rate appropriately (the learning rate passed in is the initial
        # learning rate)
        if optimizer is not None:
            self.optimizer = optimizer
        elif getattr(self, 'optimizer', None) is None:
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Remember the start time - we'll log out the total time for training later
        callbacks.on_train_begin()
        t0 = time.time()

        # Loop through `batch_size` sized subsets of the total paths and train on each
        for epoch in range(self.n_epochs):
            callbacks.on_epoch_begin(epoch)

            for step in range(self.epoch_size):
                callbacks.on_train_batch_begin(step)

                # Get a random initial spot so that the training sees a proper range even at t=0.
                # We use the same initial spot price across the batch so that all MC paths are sampled
                # from the same distribution, which is a requirement for our expected shortfall calculation.
                init_spot = self.generate_random_init_spot()
                compute_loss = lambda: self.compute_loss(init_spot)

                # Now we've got the inputs set up for the training - run the training step
                # TODO should we use a GradientTape and then call compute_loss and apply_gradients?
                self.optimizer.minimize(compute_loss, self.trainable_variables)

                logs = {'batch': step, 'size': self.batch_size, 'loss': compute_loss().numpy()}
                callbacks.on_train_batch_end(step, logs)

            logs.update(val_loss=self.test(n_paths=self.n_val_paths))
            callbacks.on_epoch_end(epoch, logs)

            if self.stop_training:
                break

        if verbose != 0:
            duration = get_duration_desc(t0)
            log.info('Total training time: %s', duration)

        callbacks.on_train_end()
        return self.history

    def test(self, n_paths, *, verbose=0):
        """Test model performance by computing the Expected Shortfall of the PNLs from a
        Monte Carlo simulation of the trading strategy represented by the model.
        
        Parameters
        ----------
        n_paths : int
            Number of paths to test against.
        verbose : int
            Verbosity, use 0 to turn off all logging.
        
        Returns
        -------
        float
            Expected Shortfall of the PNLs of the test MC simulation.
        """

        _, _, nn_pnls = self.simulate(n_paths, verbose=verbose)
        nn_es = calc_expected_shortfall(nn_pnls, self.pctile)

        return nn_es

    def simulate(self, n_paths, *, verbose=1, write_to_tensorboard=False):
        """Simulate the trading strategy and return the PNLs.
        
        Parameters
        ----------
        n_paths : int
            Number of paths to test against.
        verbose : int
            Verbosity, use 0 to turn off all logging.
        write_to_tensorboard : bool
            Whether to write to tensorboard or not.
        
        Returns
        -------
        tuple of :obj:`numpy.array`
            (unhedged pnl, Black-Scholes hedged pnl, neural network hedged pnl)
        """
        raise NotImplementedError()

    def restore(self):
        """Restore model weights from most recent checkpoint."""
        try:
            self.load_weights(self.checkpoint_prefix)
        except ValueError:
            # No checkpoint to restore from
            pass
