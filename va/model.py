"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Mark Higgins, Ben Pryke
Description: |
    Variable Annuity model.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import tensorflow as tf

import lib.black_scholes as bs
from lib.utils import get_duration_desc

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = tf.summary.create_file_writer('logs/')


def set_seed(seed=1):
    """Seed the RNGs so we get consistent results from run to run"""
    import os
    import random
    
    log.info('Using seed %d', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calc_fair_va_fee(texp, gmdb, S0, vol, lam):
    """Fair fee that the insurer receives to make the whole structure zero cost"""

    def port_value(est_fee):
        def integ(t):
            fwd = S0 * np.exp(-est_fee * t)
            opt = bs.opt_price(False, fwd, gmdb, t, vol, 0, 0)

            return opt * np.exp(-lam * t) * lam

        drift_rn = -est_fee - lam

        val = -scipy.integrate.quad(integ, 0, texp)[0]
        if drift_rn == 0:
            val += est_fee * texp
        else:
            val += est_fee * (np.exp(drift_rn * texp) - 1) / drift_rn

        return val

    fair_fee = scipy.optimize.newton(port_value, 1e-3)
    return float(fair_fee)


def compute_analytical_bs_delta(texp, start_time, lam, vol, fee, gmdb, account, spot):
    """Delta to put on against the VA portfolio"""
    
    t_fwd = texp - start_time
    n_int = max(2, int(round(0.4 * t_fwd)))
    dt_int = t_fwd / n_int
    deltas = -fee * account * np.exp(-lam * start_time) / spot * (1 - np.exp(-(fee + lam) * t_fwd)) / (fee + lam)
    for j in range(n_int):
        t_int = (j + 0.5) * dt_int  # time from current time to put expiration
        adj_strikes = gmdb * spot / account * np.exp(fee * t_int)
        d1s = (np.log(spot / adj_strikes) + vol * vol * t_int / 2) / vol / np.sqrt(t_int)
        put_deltas = scipy.stats.norm.cdf(d1s * -1) * -1 * account / spot * np.exp(-fee * t_int)
        t_s = j * dt_int
        t_e = t_s + dt_int
        deltas += (np.exp(-lam * t_s) - np.exp(-lam * t_e)) * put_deltas * np.exp(-lam * start_time)
    
    return deltas


def calc_expected_shortfall(pnls, pctile):
    """Calculate the expected shortfall across a number of paths.
    
    The option price is just that, too, since adding cash to make it zero is the minimum
    price we'd need to accept. This of course is like an "offer" price because it includes
    some risk aversion for PNL noise around the mean - but if we've got an accurate
    hedging strategy then it's not going to be much.
    
    Parameters
    ----------
    pnls : :obj:`numpy.array`
        Array of pnls for a number of paths.
    """
    
    n_pct = int((100 - pctile) / 100 * len(pnls))
    pnls = np.sort(pnls)
    price = pnls[:n_pct].mean()
    
    return price


class Hyperparams:
    n_layers = 2
    n_hidden = 50 # Number of nodes per hidden layer
    learning_rate = 5e-3 # Adam optimizer initial learning rate
    beta_1 = 0.9
    w_stddev = 0.1
    b_stddev = 1.0
    
    batch_size = 100 # Number of MC paths to include in one step of the neural network training
    n_batches = 10_000
    n_test_paths = 100_000 # Number of MC paths
    
    S0 = 1.0 # initial spot price
    mu = 0.0 # Expected upward spot drift, in years
    vol = 0.2 # Volatility
    
    texp = 5.0 # Fixed tenor to expiration, in years
    principal = 100.0 # Initial investment lump sum
    gmdb_frac = 1.
    gmdb = gmdb_frac * principal # Guaranteed minimum death benefit, floored at principal investment
    lam = 0.01 # (constant) probability of death per year
    fee = calc_fair_va_fee(texp, gmdb_frac, S0, vol, lam) # Annual fee percentage
    
    dt = 1 / 12
    n_steps = int(texp / dt) # number of time steps
    pctile = 70 # percentile for expected shortfall
    
    def __setattr__(self, name, value):
        """Ensure the fair fee is kept up to date"""
        self.__dict__[name] = value
        
        if name in ('texp', 'gmdb_frac', 'S0', 'vol', 'lam'):
            self.fee = calc_fair_va_fee(self.texp, self.gmdb_frac, self.S0, self.vol, self.lam)


class Model(Hyperparams):
    def __init__(self, **kwargs):
        """Define our NN structure; we use the same nodes in each timestep.
        
        Network inputs are spot and the time to expiration.
        Network output is delta hedge notional.
        """
        
        # Set hyperparameters
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError('Invalid hyperparameter "%s"', k)
        
        # Hidden layers
        self.b_nodes = []
        self.W_nodes = []
        
        for hidden_index in range(self.n_layers):
            b = tf.Variable(tf.random.truncated_normal([self.n_hidden], stddev=self.b_stddev), trainable=True)
            
            if hidden_index == 0:
                W = tf.Variable(tf.random.truncated_normal([2, self.n_hidden], stddev=self.w_stddev), trainable=True)
            else:
                W = tf.Variable(tf.random.truncated_normal([self.n_hidden, self.n_hidden], stddev=self.w_stddev), trainable= True)
            
            self.b_nodes.append(b)
            self.W_nodes.append(W)
        
        # Output
        # We have one output (notional of spot hedge) is a linear combination of the second
        # hidden layer node values
        self.bo = tf.Variable(tf.random.truncated_normal([1], stddev=self.b_stddev), trainable=True)
        self.Wo = tf.Variable(tf.random.truncated_normal([self.n_hidden, 1], stddev=self.w_stddev), trainable=True)
        
        # Inputs
        # Our 2 inputs are spot price and time, which are mostly determined during the MC
        # simulation except for the initial spot at time 0
        self.init_spot = tf.Variable(self.S0)
        self.nn_pnl = tf.Variable(np.zeros(self.batch_size, dtype=np.float32))
    
    @property
    def trainable_variables(self):
        return self.W_nodes + self.b_nodes + [self.Wo, self.bo]
    
    @tf.function
    def compute_hedge_delta(self, x):
        """Returns the output of the neural network at any point in time
        
        The delta size of the position required to hedge the option.
        """
        h = x
        
        for hidden_index in range(self.n_layers):
            h = tf.nn.relu(tf.matmul(h, self.W_nodes[hidden_index]) + self.b_nodes[hidden_index])
        
        y = tf.matmul(h, self.Wo) + self.bo
        
        return -y ** 2
    
    @tf.function
    def compute_pnl(self):
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
        
        pnl = tf.zeros(self.batch_size, dtype=tf.float32) # Account values are not part of insurer pnl
        spot = tf.zeros(self.batch_size, dtype=tf.float32) + self.init_spot
        log_spot = tf.zeros(self.batch_size, dtype=tf.float32)
        account = tf.zeros(self.batch_size, dtype=tf.float32) + self.principal # Every path represents an infinite number of accounts
        
        # Run through the MC sim, generating path values for spots along the way
        for time_index in tf.range(self.n_steps, dtype=tf.float32):
            """Compute updates at start of interval"""
            t = time_index * self.dt
            
            # Retrieve the neural network output, treating it as the delta hedge notional
            # at the start of the timestep. In the risk neutral limit, Black-Scholes is equivallent
            # to the minimising expected shortfall. Therefore, by minimising expected shortfall as
            # our loss function, the output of the network is trained to approximate Black-Scholes delta.
            input_time = tf.fill([self.batch_size], t)
            inputs = tf.stack([spot, input_time], 1)
            delta = self.compute_hedge_delta(inputs)[:, 0]
            delta *= tf.minimum(tf.math.exp(-0.01 * delta), 1.)
            delta *= (1 - tf.math.exp(-self.lam * (self.texp - t))) * self.principal
            
            account = self.principal * spot / self.S0 * tf.math.exp(-self.fee * t)
            fee = self.fee * self.dt * account * tf.math.exp(-self.lam * t)
            payout = self.lam * self.dt * tf.maximum(self.gmdb - account, 0) * tf.math.exp(-self.lam * t)
            inc_pnl = fee - payout
            
            """Compute updates at end of interval"""
            # The stochastic process is defined in the real world measure, not the risk neutral one.
            # The process is:
            #     dS = mu S dt + vol S dz_s
            # where the model parameters are mu and vol. mu is the (real world) drift of the asset price S.
            rs = tf.random.normal([self.batch_size], 0, self.dt ** 0.5)
            log_spot += (self.mu - self.vol * self.vol / 2.) * self.dt + self.vol * rs
            new_spot = self.init_spot * tf.math.exp(log_spot)
            spot_change = new_spot - spot
            
            # Update the PNL and dynamically delta hedge
            pnl += inc_pnl
            pnl += delta * spot_change
            
            # Remember values for the next step
            spot = new_spot
        
        return pnl
    
    @tf.function
    def compute_loss(self):
        """Use expected shortfall for the appropriate percentile as the loss function.
        
        Note that we do *not* expect this to minimize to zero.
        """
        
        pnl = self.compute_pnl()
        n_pct = int((100 - self.pctile) / 100 * self.batch_size)
        pnl_past_cutoff = tf.nn.top_k(-pnl, n_pct)[0]
        return tf.reduce_mean(pnl_past_cutoff)
    
    @tf.function
    def compute_mean_pnl(self):
        """Mean PNL for debugging purposes"""
        pnl = self.compute_pnl()
        return tf.reduce_mean(pnl)


def train(model):
    """We'll train the network by running an MC simulation, batching up the paths into groups of batch_size paths"""
    
    n_paths = model.batch_size * model.n_batches
    log.info('Training on %d paths over %d batches', n_paths, model.n_batches)
    
    # Use the Adam optimizer, which is gradient descent which also evolves
    # the learning rate appropriately (the learning rate passed in is the initial`
    # learning rate)
    optimizer = tf.keras.optimizers.Adam(model.learning_rate, model.beta_1)
    compute_loss = lambda: model.compute_loss()
    
    # Remember the start time - we'll log out the total time for training later
    t0 = time.time()
    losses = []
    # init_spots = np.linspace(0.3, 3, 1000, endpoint=True, dtype=np.float32)[::-1]
    
    # Loop through the `batch_size`-sized subsets of the total paths and train on each
    for batch in range(model.n_batches):
        # if batch == 500:
        #     optimizer = tf.keras.optimizers.Adam(1e-3)
        # Get a random initial spot so that the training sees a proper range even at t=0.
        # We use the same initial spot price across the batch so that all MC paths are sampled
        # from the same distribution, which is a requirement for our expected shortfall calculation.
        init_spot = model.S0 * np.exp(-model.vol * model.vol * model.texp / 2. + np.random.normal(0, 2. * model.vol * model.texp ** 0.5))
        # init_spot = init_spots[batch % 1000]
        model.init_spot.assign(init_spot)
        
        # Now we've got the inputs set up for the training - run the training step
        optimizer.minimize(compute_loss, model.trainable_variables)
        loss = compute_loss().numpy()
        losses.append(loss)
        
        # Log some stats as we train
        if batch % 100 == 0:
            mean_pnl = model.compute_mean_pnl().numpy()
            duration = get_duration_desc(t0)
            log.info('Batch %04d (%s): loss % .5f, mean % .5f, init spot % .5f', batch, duration, loss, mean_pnl, init_spot)
    
    duration = get_duration_desc(t0)
    log.info('Total training time: %s', duration)
    
    return losses


def test(model):
    """Test model performance by comparing with analytically computed Black-Scholes hedges."""
    
    n_paths = model.n_test_paths
    log.info('Testing on %d paths', n_paths)
    
    log_spot = np.zeros(n_paths, dtype=np.float32)
    nn_pnls = np.zeros(n_paths, dtype=np.float32)
    bs_pnls = np.zeros(n_paths, dtype=np.float32)
    spot = np.zeros(n_paths, dtype=np.float32) + model.S0
    account = np.zeros(n_paths, dtype=np.float32) + model.principal # Every path represents an infinite number of accounts
    
    # Run through the MC sim, generating path values for spots along the way. This is just like a regular MC
    # sim to price a derivative - except that the price is *not* the expected value - it's the loss function
    # value. That handles both the conversion from real world to "risk neutral" and unhedgeable risk due to
    # eg discrete hedging (which is the only unhedgeable risk in this example, but there could be anything generally).
    for time_index in range(model.n_steps):
        """Compute updates at start of interval"""
        t = time_index * model.dt
        
        # Compute deltas
        input_time = tf.constant([t] * n_paths)
        nn_input = tf.stack([spot, input_time], 1)
        nn_delta = model.compute_hedge_delta(nn_input)[:, 0].numpy()
        nn_delta = np.minimum(nn_delta, 0) # pylint: disable=assignment-from-no-return
        nn_delta *= (1 - np.exp(-model.lam * (model.texp - t))) * model.principal
        
        bs_delta = compute_analytical_bs_delta(model.texp, t, model.lam, model.vol, model.fee, model.gmdb, account, spot)
        
        # Compute step updates
        account = model.principal * spot / model.S0 * np.exp(-model.fee * t)
        fee = model.fee * model.dt * account * np.exp(-model.lam * t)
        payout = model.lam * model.dt * np.maximum(model.gmdb - account, 0) * np.exp(-model.lam * t)
        inc_pnl = fee - payout
        
        """Compute updates at end of interval"""
        # Advance MC sim
        rs = np.random.normal(0, model.dt ** 0.5, size=n_paths)
        log_spot += (model.mu - model.vol * model.vol / 2.) * model.dt + model.vol * rs
        new_spot = model.S0 * np.exp(log_spot)
        spot_change = new_spot - spot
        
        # Update the PNL and dynamically delta hedge
        nn_pnls += inc_pnl + nn_delta * spot_change
        bs_pnls += inc_pnl + bs_delta * spot_change
        
        # Remember values for the next step
        spot = new_spot
    
    nn_price = calc_expected_shortfall(nn_pnls, model.pctile)
    bs_price = calc_expected_shortfall(bs_pnls, model.pctile)
    
    return abs(bs_price - nn_price)


def compute_expected_shortfalls(model):
    """Calculate the option price from this optimal hedging strategy. We calculate the expected shortfall; the 
    option price is just that, since adding cash to make it zero is the minimum price we'd need to accept.
    This of course is like an "offer" price because it includes some risk aversion for PNL noise around the 
    mean - but if we've got an accurate hedging strategy then it's not going to be much.
    """
    
    t0 = time.time()
    n_paths = model.n_test_paths
    
    log_spot = np.zeros(n_paths, dtype=np.float32)
    uh_pnls = np.zeros(n_paths, dtype=np.float32)
    nn_pnls = np.zeros(n_paths, dtype=np.float32)
    bs_pnls = np.zeros(n_paths, dtype=np.float32)
    spot = np.zeros(n_paths, dtype=np.float32) + model.S0
    account = np.zeros(n_paths, dtype=np.float32) + model.principal # Every path represents an infinite number of accounts
    data = []
    # Run through the MC sim, generating path values for spots along the way. This is just like a regular MC
    # sim to price a derivative - except that the price is *not* the expected value - it's the loss function
    # value. That handles both the conversion from real world to "risk neutral" and unhedgeable risk due to
    # eg discrete hedging (which is the only unhedgeable risk in this example, but there could be anything generally).
    for time_index in range(model.n_steps):
        """Compute updates at start of interval"""
        t = time_index * model.dt
        
        # Compute deltas
        input_time = tf.constant([t] * n_paths)
        nn_input = tf.stack([spot, input_time], 1)
        nn_delta = model.compute_hedge_delta(nn_input)[:, 0].numpy()
        nn_delta = np.minimum(nn_delta, 0) # pylint: disable=assignment-from-no-return
        nn_delta *= (1 - np.exp(-model.lam * (model.texp - t))) * model.principal
        
        bs_delta = compute_analytical_bs_delta(model.texp, t, model.lam, model.vol, model.fee, model.gmdb, account, spot)
        data.append(abs(np.mean(nn_delta - bs_delta)))
        
        # Compute step updates
        account = model.principal * spot / model.S0 * np.exp(-model.fee * t)
        fee = model.fee * model.dt * account * np.exp(-model.lam * t)
        payout = model.lam * model.dt * np.maximum(model.gmdb - account, 0) * np.exp(-model.lam * t)
        inc_pnl = fee - payout
        
        """Compute updates at end of interval"""
        # Advance MC sim
        rs = np.random.normal(0, model.dt ** 0.5, n_paths)
        log_spot += (model.mu - model.vol * model.vol / 2.) * model.dt + model.vol * rs
        new_spot = model.S0 * np.exp(log_spot)
        spot_change = new_spot - spot
        
        # Update the PNL and dynamically delta hedge
        uh_pnls += inc_pnl
        nn_pnls += inc_pnl + nn_delta * spot_change
        bs_pnls += inc_pnl + bs_delta * spot_change
        
        # Remember values for the next step
        spot = new_spot
        
        log.info('%.4f years - delta: mean % .5f, std % .5f; spot: mean % .5f, std % .5f', t, nn_delta.mean(), nn_delta.std(), spot.mean(), spot.std())
        
        with writer.as_default():
            tf.summary.histogram('nn_delta', nn_delta, step=time_index)
            tf.summary.histogram('bs_delta', bs_delta, step=time_index)
            tf.summary.histogram('nn_pnls', nn_pnls, step=time_index)
            tf.summary.histogram('bs_pnls', bs_pnls, step=time_index)
            tf.summary.histogram('inc_pnl', inc_pnl, step=time_index)
            tf.summary.histogram('log_spot', log_spot, step=time_index)
            tf.summary.histogram('spot', spot, step=time_index)
            tf.summary.histogram('fee', fee, step=time_index)
            tf.summary.histogram('payout', payout, step=time_index)
    
    writer.flush()
    
    uh_price = calc_expected_shortfall(uh_pnls, model.pctile)
    nn_price = calc_expected_shortfall(nn_pnls, model.pctile)
    bs_price = calc_expected_shortfall(bs_pnls, model.pctile)
    duration = get_duration_desc(t0)
    
    log.info('Unhedged ES      = % .5f (mean % .5f, std % .5f)', uh_price, np.mean(uh_pnls), np.std(uh_pnls))
    log.info('Deep hedging ES  = % .5f (mean % .5f, std % .5f)', nn_price, np.mean(nn_pnls), np.std(nn_pnls))
    log.info('Black-Scholes ES = % .5f (mean % .5f, std % .5f)', bs_price, np.mean(bs_pnls), np.std(bs_pnls))
    log.info('Simulation time: %s', duration)
    
    return uh_pnls, bs_pnls, nn_pnls
