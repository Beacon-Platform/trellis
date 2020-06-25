# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Variable Annuity model."""

import logging
import time

import numpy as np
import tensorflow as tf

from trellis.models.base import Model, HyperparamsBase
from trellis.models.variable_annuity import analytics
from trellis.models.utils import depends_on
from trellis.random_processes import gbm
from trellis.utils import calc_expected_shortfall, get_duration_desc

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Hyperparams(HyperparamsBase):
    n_layers = 2
    n_hidden = 50  # Number of nodes per hidden layer
    w_std = 0.05  # Initialisation std of the weights
    b_std = 0.05  # Initialisation std of the biases

    learning_rate = 5e-3
    batch_size = 100  # Number of MC paths per batch
    epoch_size = 100  # Number of batches per epoch
    n_epochs = 100  # Number of epochs to train for
    n_val_paths = 50_000  # Number of paths to validate against
    n_test_paths = 100_000  # Number of paths to test against

    S0 = 1.0  # initial spot price
    mu = 0.0  # Expected upward spot drift, in years
    vol = 0.2  # Volatility

    texp = 5.0  # Fixed tenor to expiration, in years
    principal = 100.0  # Initial investment lump sum
    gmdb_frac = 1.0
    gmdb = gmdb_frac * principal  # Guaranteed minimum death benefit, floored at principal investment
    lam = 0.01  # (constant) probability of death per year

    dt = 1 / 12  # Timesteps per year
    n_steps = int(texp / dt)  # Number of time steps
    pctile = 70  # Percentile for expected shortfall

    @property
    @depends_on('texp', 'gmdb_frac', 'S0', 'vol', 'lam')
    def fee(self):
        """Annual fee percentage"""
        return analytics.calc_fair_fee(self.texp, self.gmdb_frac, self.S0, self.vol, self.lam)

    @property
    def critical_fields(self):
        """Tuple of parameters that uniquely define the model."""
        return (
            self.n_layers,
            self.n_hidden,
            self.w_std,
            self.b_std,
            self.learning_rate,
            self.batch_size,
            self.S0,
            self.mu,
            self.vol,
            self.texp,
            self.principal,
            self.lam,
            self.dt,
            self.pctile,
        )


class VariableAnnuity(Model, Hyperparams):
    def __init__(self, **kwargs):
        """Define our NN structure; we use the same nodes in each timestep.
        
        Network inputs are spot and the time to expiration.
        Network output is delta hedge notional.
        """

        Hyperparams.__init__(self, **kwargs)
        Model.__init__(self)

        # Hidden layers
        for _ in range(self.n_layers):
            self.add(
                tf.keras.layers.Dense(
                    units=self.n_hidden,
                    activation='relu',
                    kernel_initializer=tf.initializers.TruncatedNormal(stddev=self.w_std),
                    bias_initializer=tf.initializers.TruncatedNormal(stddev=self.b_std),
                )
            )

        # Output
        # We have one output (notional of spot hedge)
        self.add(
            tf.keras.layers.Dense(
                units=1,
                activation='linear',
                kernel_initializer=tf.initializers.TruncatedNormal(stddev=self.w_std),
                bias_initializer=tf.initializers.TruncatedNormal(stddev=self.b_std),
            )
        )

        # Inputs
        # Our 2 inputs are spot price and time, which are mostly determined during the MC
        # simulation except for the initial spot at time 0
        self.build((None, 2))

    @tf.function
    def compute_hedge_delta(self, x):
        """Returns the output of the neural network at any point in time.
        
        The delta size of the position required to hedge the option.
        """
        return -self.call(x) ** 2

    @tf.function
    def compute_pnl(self, init_spot):
        """On each run of the training, we'll run a MC simulatiion to calculate the PNL distribution
        across the `batch_size` paths. PNL integrated along a given path is the sum of the
        option payoff at the end and the realized PNL from the hedges; that is, for a given path j,
        
            PNL[j] = Sum[delta_s[i-1,j] * (S[i,j] - S[i-1,j]), {i, 1, N}] + Payoff(S[N,j])
        
        where 
            delta_s[i,j] = spot hedge notional at time index i for path j
            S[i,j]       = spot at time index i for path j
            N            = total # of time steps from start to option expiration
            Payoff(S)    = at-expiration option payoff (ie max(S-K,0) for a call option and max(K-S,0) for a put)
        
        We can define the PNL incrementally along a path like
        
            pnl[i,j] = pnl[i-1,j] + delta_s[i-1,j] * (S[i,j] - S[i-1,j])
        
        where pnl[0,j] == 0 for every path. Then the integrated path PNL defined above is
        
            PNL[j] = pnl[N,j] + Payoff(S[N],j)
        
        So we build a tensorflow graph to calculate that integrated PNL for a given
        path. Then we'll define a loss function (given a set of path PNLs) equal to the
        expected shortfall of the integrated PNLs nfor each path in a batch. Make sure
        we're short the option so that (absent hedging) there's a -ve PNL.
        """

        pnl = tf.zeros(self.batch_size, dtype=tf.float32)  # Account values are not part of insurer pnl
        spots = gbm(init_spot, self.mu, self.vol, self.dt, self.n_steps, self.batch_size)  # Defined in the real world measure
        account = tf.zeros(self.batch_size, dtype=tf.float32) + self.principal  # Every path represents an infinite number of accounts

        # Run through the MC sim, generating path values for spots along the way
        for time_index in tf.range(self.n_steps):
            # 1. Compute updates at start of interval
            t = tf.cast(time_index, dtype=tf.float32) * self.dt
            spot = spots[time_index]

            # Retrieve the neural network output, treating it as the delta hedge notional
            # at the start of the timestep. In the risk neutral limit, Black-Scholes is equivallent
            # to the minimising expected shortfall. Therefore, by minimising expected shortfall as
            # our loss function, the output of the network is trained to approximate Black-Scholes delta.
            input_time = tf.fill([self.batch_size], t)
            inputs = tf.stack([spot, input_time], 1)
            delta = self.compute_hedge_delta(inputs)[:, 0]
            delta *= tf.minimum(tf.math.exp(-0.01 * delta), 1.0)
            delta *= (1 - tf.math.exp(-self.lam * (self.texp - t))) * self.principal

            account = self.principal * spot / self.S0 * tf.math.exp(-self.fee * t)
            fee = self.fee * self.dt * account * tf.math.exp(-self.lam * t)
            payout = self.lam * self.dt * tf.maximum(self.gmdb - account, 0) * tf.math.exp(-self.lam * t)
            inc_pnl = fee - payout

            # 2. Compute updates at end of interval
            # Delta hedge and record the incremental pnl changes over the interval
            spot_change = spots[time_index + 1] - spot
            pnl += inc_pnl + delta * spot_change

        return pnl

    @tf.function
    def compute_loss(self, init_spot):
        """Use expected shortfall for the appropriate percentile as the loss function.
        
        Note that we do *not* expect this to minimize to zero.
        """
        # TODO move to losses module as ExpectedShortfall?

        pnl = self.compute_pnl(init_spot)
        n_pct = int((100 - self.pctile) / 100 * self.batch_size)
        pnl_past_cutoff = tf.nn.top_k(-pnl, n_pct)[0]
        return tf.reduce_mean(pnl_past_cutoff)

    @tf.function
    def compute_mean_pnl(self, init_spot):
        """Mean PNL for debugging purposes"""
        pnl = self.compute_pnl(init_spot)
        return tf.reduce_mean(pnl)

    @tf.function
    def generate_random_init_spot(self):
        # TODO does this belong here?
        r = tf.random.normal((1,), 0, 2.0 * self.vol * self.texp ** 0.5)[0]
        return self.S0 * tf.exp(-self.vol * self.vol * self.texp / 2.0 + r)

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

        t0 = time.time()

        if write_to_tensorboard:
            writer = tf.summary.create_file_writer('logs/')

        spots = gbm(self.S0, self.mu, self.vol, self.dt, self.n_steps, n_paths).numpy()
        uh_pnls = np.zeros(n_paths, dtype=np.float32)
        nn_pnls = np.zeros(n_paths, dtype=np.float32)
        bs_pnls = np.zeros(n_paths, dtype=np.float32)
        account = np.zeros(n_paths, dtype=np.float32) + self.principal  # Every path represents an infinite number of accounts

        # Run through the MC sim, generating path values for spots along the way. This is just like a regular MC
        # sim to price a derivative - except that the price is *not* the expected value - it's the loss function
        # value. That handles both the conversion from real world to "risk neutral" and unhedgeable risk due to
        # eg discrete hedging (which is the only unhedgeable risk in this example, but there could be anything generally).
        for time_index in range(self.n_steps):
            # 1. Compute updates at start of interval
            t = time_index * self.dt
            spot = spots[time_index]

            # Compute deltas
            input_time = tf.constant([t] * n_paths)
            nn_input = tf.stack([spot, input_time], 1)
            nn_delta = self.compute_hedge_delta(nn_input)[:, 0].numpy()
            nn_delta = np.minimum(nn_delta, 0)  # pylint: disable=assignment-from-no-return
            nn_delta *= (1 - np.exp(-self.lam * (self.texp - t))) * self.principal

            bs_delta = analytics.calc_delta(self.texp, t, self.lam, self.vol, self.fee, self.gmdb, account, spot)

            # Compute step updates
            account = self.principal * spot / self.S0 * np.exp(-self.fee * t)
            fee = self.fee * self.dt * account * np.exp(-self.lam * t)
            payout = self.lam * self.dt * np.maximum(self.gmdb - account, 0) * np.exp(-self.lam * t)
            inc_pnl = fee - payout

            # 2. Compute updates at end of interval
            # Record incremental pnl changes over the interval
            spot_change = spots[time_index + 1] - spot
            uh_pnls += inc_pnl
            nn_pnls += inc_pnl + nn_delta * spot_change
            bs_pnls += inc_pnl + bs_delta * spot_change

            if verbose != 0:
                log.info(
                    '%.4f years - delta: mean % .5f, std % .5f; spot: mean % .5f, std % .5f',
                    t,
                    nn_delta.mean(),
                    nn_delta.std(),
                    spot.mean(),
                    spot.std(),
                )

            if write_to_tensorboard:
                with writer.as_default():
                    tf.summary.histogram('nn_delta', nn_delta, step=time_index)
                    tf.summary.histogram('bs_delta', bs_delta, step=time_index)
                    tf.summary.histogram('uh_pnls', uh_pnls, step=time_index)
                    tf.summary.histogram('nn_pnls', nn_pnls, step=time_index)
                    tf.summary.histogram('bs_pnls', bs_pnls, step=time_index)
                    tf.summary.histogram('spot', spot, step=time_index)
                    tf.summary.histogram('fee', fee, step=time_index)
                    tf.summary.histogram('payout', payout, step=time_index)
                    tf.summary.histogram('inc_pnl', inc_pnl, step=time_index)

        if write_to_tensorboard:
            writer.flush()

        if verbose != 0:
            duration = get_duration_desc(t0)
            log.info('Simulation time: %s', duration)

        return uh_pnls, bs_pnls, nn_pnls
