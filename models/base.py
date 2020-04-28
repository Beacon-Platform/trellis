# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Base classes for models."""

import logging
import os
import time

from tensorflow.python.keras.callbacks import CallbackList # pylint: disable=no-name-in-module, import-error
from tensorflow.python.keras.callbacks import configure_callbacks # pylint: disable=no-name-in-module, import-error
import tensorflow as tf

from utils import get_duration_desc

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ROOT_CHECKPOINT_DIR = './checkpoints/'


class HyperparamsBase:
    root_checkpoint_dir = ROOT_CHECKPOINT_DIR
    
    learning_rate = 5e-3
    batch_size = 100
    n_batches = 10_000
    n_test_paths = 100_000
    
    def __init__(self, **kwargs):
        # Set hyperparameters
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError('Invalid hyperparameter "%s"', k)
    
    @property
    def checkpoint_directory(self):
        """Directory in which to save checkpoint files."""
        raise NotImplementedError()
    
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
        """We'll train the network by running an MC simulation, batching up the paths into groups of batch_size paths"""
        
        # Note that these aren't real epochs, but are used for monitoring val_loss (which is also different from loss...)
        n_paths = self.batch_size * self.n_batches
        n_steps_per_epoch = 100
        n_epochs = int(self.n_batches / n_steps_per_epoch)
        
        if verbose != 0:
            log.info('Training on %d paths over %d batches in %d epochs', n_paths, self.n_batches, n_epochs)
        
        if not isinstance(callbacks, CallbackList):
            callbacks = list(callbacks or [])
            callbacks = configure_callbacks(
                callbacks,
                self,
                do_validation=True,
                batch_size=self.batch_size,
                epochs=n_epochs,
                steps_per_epoch=n_steps_per_epoch,
                verbose=verbose,
            )
        
        # Use the Adam optimizer, which is gradient descent which also evolves
        # the learning rate appropriately (the learning rate passed in is the initial`
        # learning rate)
        if optimizer is not None:
            self.optimizer = optimizer
        elif getattr(self, 'optimizer', None) is None:
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        # Remember the start time - we'll log out the total time for training later
        callbacks.on_train_begin()
        t0 = time.time()
        
        # Loop through the `batch_size`-sized subsets of the total paths and train on each
        for epoch in range(n_epochs):
            callbacks.on_epoch_begin(epoch)
            
            for step in range(n_steps_per_epoch):
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
            
            logs.update(val_loss=self.test())
            callbacks.on_epoch_end(epoch, logs)
            
            if self.stop_training:
                break
        
        if verbose != 0:
            duration = get_duration_desc(t0)
            log.info('Total training time: %s', duration)
        
        callbacks.on_train_end()
        return self.history
    
    def restore(self):
        """Restore model weights from most recent checkpoint."""
        try:
            self.load_weights(self.checkpoint_prefix).expect_partial()
        except ValueError:
            # No checkpoints to restore from
            pass
