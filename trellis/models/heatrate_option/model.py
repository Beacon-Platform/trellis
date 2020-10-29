# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins, Amine Benchrifa

"""Heat Rate Option model."""

import logging
import time

import numpy as np
import tensorflow as tf

from models.base import Model, HyperparamsBase
import models.heatrate_option.analytics as analytics
from utils import calc_expected_shortfall, get_duration_desc


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Hyperparams(HyperparamsBase):
    n_layers = 2
    n_hidden = 50  # Number of nodes per hidden layer
    w_std = 0.1  # Initialisation std of the weights
    b_std = 0.05  # Initialisation std of the biases

    learning_rate = 1e-3
    batch_size = 100  # Number of MC paths per batch
    epoch_size = 100  # Number of batches per epoch
    n_epochs = 100  # Number of epochs to train for
    n_val_paths = 50_000  # Number of paths to validate against
    n_test_paths = 100_000  # Number of paths to test against

    SP0 = 45.0  # initial power price
    SG0 = 12.0  # initial gas price
    H = 2.35  # heat rate
    mu_P = 0.0025  # Expected upward spot(power) drift, in years
    vol_P = 0.35  # Volatility (power)
    mu_G = 0.0025  # Expected upward spot(gas) drift, in years
    vol_G = 0.6  # Volatility (gas)
    rho = 0.8  # Correlation between power and gas prices

    texp = 1  # Fixed tenor to expiration, in years
    K = 15.0  # option strike price
    is_call = True  # True: call option; False: put option
    is_buy = False  # True: buying a call/put; False: selling a call/put
    phi = 1 if is_call else -1  # Call or put
    psi = 1 if is_buy else -1  # Buy or sell

    dt = 1 / 260  # Timesteps per year
    n_steps = int(texp / dt)  # Number of time steps
    pctile = 70  # Percentile for expected shortfall

    def __setattr__(self, name, value):
        """Ensure the fair fee is kept up to date"""
        # TODO Can we use non-trainable Variables to form a dependency tree so we don't need to update these without losing functionality?
        self.__dict__[name] = value

        if name == 'is_call':
            self.phi = 1 if self.is_call else -1

        if name == 'is_buy':
            self.psi = 1 if self.is_buy else -1

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
            self.epoch_size,
            self.n_epochs,
            self.SP0,
            self.SG0,
            self.H,
            self.mu_P,
            self.vol_P,
            self.mu_G,
            self.vol_G,
            self.rho,
            self.texp,
            self.K,
            self.is_call,
            self.is_buy,
            self.dt,
            self.pctile,
        )


class HeatrateOption(Model, Hyperparams):
    def __init__(self, **kwargs):
        """Define our NN structure; we use the same nodes in each timestep"""

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
        # We have two outputs (notional of spot hedge)
        self.add(
            tf.keras.layers.Dense(
                units=2,
                activation='linear',
                kernel_initializer=tf.initializers.TruncatedNormal(stddev=self.w_std),
                bias_initializer=tf.initializers.TruncatedNormal(stddev=self.b_std),
            )
        )

        # Inputs
        # Our 3 inputs are power price, gas price and time, which are mostly determined during the MC
        # simulation except for the initial spot at time 0
        self.build((None, 3))

    @tf.function
    def compute_hedge_delta(self, x):
        """Returns the output of the neural network at any point in time.

        The delta size of the position required to hedge the option.
        """
        return self.call(x)

    @tf.function
    def compute_pnl(self, init_spot):
        """On each run of the training, we'll run a MC simulatiion to calculate the PNL distribution
        across the `batch_size` paths. PNL integrated along a given path is the sum of the
        option payoff at the end and the realized PNL from the hedges; that is, for a given path j,

            PNL[j] = Sum[delta_p[i-1,j] * (S[i,j] - S[i-1,j])-abs(delta_g[i-1,j])*(S[i,j] - S[i-1,j]), {i, 1, N}] + Payoff(S[N,j])

        where
            delta_p[i,j] = power spot hedge notional at time index i for path j
            delta_g[i,j] = gas spot hedge notional at time index i for path j
            S[i,j]       = spot at time index i for path j
            N            = total # of time steps from start to option expiration
            Payoff(S)    = at-expiration option payoff

        We can define the PNL incrementally along a path like

            pnl[i,j] = pnl[i-1,j] + delta_p[i-1,j] * (S[i,j] - S[i-1,j])-abs(delta_g[i-1,j])*(S[i,j] - S[i-1,j])

        where pnl[0,j] == 0 for every path. Then the integrated path PNL defined above is

            PNL[j] = pnl[N,j] + Payoff(S[N],j)

        So we build a tensorflow graph to calculate that integrated PNL for a given
        path. Then we'll define a loss function (given a set of path PNLs) equal to the
        expected shortfall of the integrated PNLs for each path in a batch. Make sure
        we're short the option so that (absent hedging) there's a -ve PNL.
        """

        init_spot_power, init_spot_gas = init_spot

        pnl = tf.zeros(self.batch_size, dtype=tf.float32)

        spot_power = tf.zeros(self.batch_size, dtype=tf.float32) + init_spot_power
        spot_gas = tf.zeros(self.batch_size, dtype=tf.float32) + init_spot_gas

        log_spot_power = tf.zeros(self.batch_size, dtype=tf.float32)
        log_spot_gas = tf.zeros(self.batch_size, dtype=tf.float32)

        # Run through the MC sim, generating path values for spots along the way
        for time_index in tf.range(self.n_steps, dtype=tf.float32):
            """Compute updates at start of interval"""
            t = time_index * self.dt

            # Retrieve the neural network output, treating it as the delta hedge notional
            # at the start of the timestep. In the risk neutral limit, Black-Scholes is equivallent
            # to the minimising expected shortfall. Therefore, by minimising expected shortfall as
            # our loss function, the output of the network is trained to approximate Black-Scholes delta.
            input_time = tf.fill([self.batch_size], t)
            inputs = tf.stack([spot_power, spot_gas, input_time], 1)

            delta_power = self.compute_hedge_delta(inputs)[:, 0]
            delta_gas = self.compute_hedge_delta(inputs)[:, 1]

            """Compute updates at end of interval"""

            rs1 = tf.random.normal([self.batch_size], 0, self.dt ** 0.5)
            rs2 = tf.random.normal([self.batch_size], 0, self.dt ** 0.5)

            w1 = rs1
            w2 = self.rho * w1 + ((1 - self.rho * self.rho) ** 0.5) * rs2

            log_spot_power += (self.mu_P - self.vol_P * self.vol_P / 2.0) * self.dt + self.vol_P * w1
            log_spot_gas += (self.mu_G - self.vol_G * self.vol_G / 2.0) * self.dt + self.vol_G * w2

            new_spot_power = init_spot_power * tf.math.exp(log_spot_power)
            new_spot_gas = init_spot_gas * tf.math.exp(log_spot_gas)

            spot_change_power = new_spot_power - spot_power
            spot_change_gas = new_spot_gas - spot_gas

            # Update the PNL and dynamically delta hedge
            pnl += delta_power * spot_change_power + delta_gas * spot_change_gas

            # Remember values for the next step
            spot_power = new_spot_power
            spot_gas = new_spot_gas

        # Calculate the final payoff
        payoff = self.psi * tf.maximum(self.phi * (spot_power - self.H * spot_gas - self.K), 0)
        pnl += payoff  # Note we sell the option here

        return pnl

    @tf.function
    def compute_loss(self, init_spot):
        """Use expected shortfall for the appropriate percentile as the loss function.

        Note that we do *not* expect this to minimize to zero.
        """
        # TODO move to losses module as ExpectedShortfall?
        init_spot_power, init_spot_gas = init_spot

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
        r_P, r_G = tf.random.normal((2,), 0, 2.0 * self.vol_P * self.texp ** 0.5)

        S_P = self.SP0 * tf.exp(-self.vol_P * self.vol_P * self.texp / 2.0 + r_P)
        S_G = self.SG0 * tf.exp(-self.vol_G * self.vol_G * self.texp / 2.0 + r_G)

        return (S_P, S_G)

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

        log_spot_power = np.zeros(n_paths, dtype=np.float32)
        log_spot_gas = np.zeros(n_paths, dtype=np.float32)

        uh_pnls = np.zeros(n_paths, dtype=np.float32)
        nn_pnls = np.zeros(n_paths, dtype=np.float32)
        bs_pnls = np.zeros(n_paths, dtype=np.float32)

        spot_power = np.zeros(n_paths, dtype=np.float32) + self.SP0
        spot_gas = np.zeros(n_paths, dtype=np.float32) + self.SG0

        # Run through the MC sim, generating path values for spots along the way. This is just like a regular MC
        # sim to price a derivative - except that the price is *not* the expected value - it's the loss function
        # value. That handles both the conversion from real world to "risk neutral" and unhedgeable risk due to
        # eg discrete hedging (which is the only unhedgeable risk in this example, but there could be anything generally).
        for time_index in range(self.n_steps):
            """Compute updates at start of interval"""
            t = time_index * self.dt

            input_time = tf.constant([t] * n_paths)
            nn_input = tf.stack([spot_power, spot_gas, input_time], 1)
            nn_deltas= self.compute_hedge_delta(nn_input)
            nn_delta_power = nn_deltas[:, 0].numpy()
            nn_delta_gas = nn_deltas[:, 1].numpy()
            
            bs_deltas = analytics.calc_opt_delta(
                self.is_call, spot_power, spot_gas, self.K, self.H, self.texp - t, self.vol_P, self.vol_G, 0, self.rho
            )

            bs_delta_power = -self.psi*bs_deltas[0]

            bs_delta_gas = -self.psi*bs_deltas[1]

            """Compute updates at end of interval"""
            # Advance MC sim
            rs1 = np.random.normal(0, self.dt ** 0.5, n_paths)
            rs2 = np.random.normal(0, self.dt ** 0.5, n_paths)

            w1 = rs1
            w2 = self.rho * w1 + ((1 - self.rho * self.rho) ** 0.5) * rs2

            log_spot_power += (self.mu_P - self.vol_P * self.vol_P / 2.0) * self.dt + self.vol_P * w1
            log_spot_gas += (self.mu_G - self.vol_G * self.vol_G / 2.0) * self.dt + self.vol_G * w2

            new_spot_power = self.SP0 * np.exp(log_spot_power)
            new_spot_gas = self.SG0 * np.exp(log_spot_gas)

            spot_change_power = new_spot_power - spot_power
            spot_change_gas = new_spot_gas - spot_gas

            # Get the PNLs of the hedges over the interval
            nn_pnls += nn_delta_power * spot_change_power + nn_delta_gas * spot_change_gas
            bs_pnls += bs_delta_power * spot_change_power + bs_delta_gas * spot_change_gas

            # Remember stuff for the next time step
            spot_power = new_spot_power
            spot_gas = new_spot_gas

            if verbose != 0:
                log.info(
                    '%.4f years - delta_power: mean % .4f,std % .4f; delta_gas: mean % .4f,std % .4f; spot_power: mean % .4f, std % .4f; spot_gas: mean % .4f, std % .4f',
                    t,
                    nn_delta_power.mean(),
                    nn_delta_power.std(),
                    nn_delta_gas.mean(),
                    nn_delta_gas.std(),
                    spot_power.mean(),
                    spot_power.std(),
                    spot_gas.mean(),
                    spot_gas.std(),
                )

            if write_to_tensorboard:
                with writer.as_default():
                    tf.summary.histogram('nn_delta_power', nn_delta_power, step=time_index)
                    tf.summary.histogram('bs_delta_power', bs_delta_power, step=time_index)
                    tf.summary.histogram('nn_delta_gas', nn_delta_gas, step=time_index)
                    tf.summary.histogram('bs_delta_gas', bs_delta_gas, step=time_index)
                    tf.summary.histogram('nn_pnls', nn_pnls, step=time_index)
                    tf.summary.histogram('bs_pnls', bs_pnls, step=time_index)
                    tf.summary.histogram('log_spot_power', log_spot_power, step=time_index)
                    tf.summary.histogram('spot_power', spot_power, step=time_index)
                    tf.summary.histogram('log_spot_gas', log_spot_gas, step=time_index)
                    tf.summary.histogram('spot_gas', spot_gas, step=time_index)

        # Compute the payoff and some metrics

        payoff = self.psi * np.maximum(self.phi * (spot_power - self.H * spot_gas - self.K), 0)
        uh_pnls += payoff
        nn_pnls += payoff
        bs_pnls += payoff

        if write_to_tensorboard:
            writer.flush()

        if verbose != 0:
            duration = get_duration_desc(t0)
            log.info('Simulation time: %s', duration)

        return uh_pnls, bs_pnls, nn_pnls
