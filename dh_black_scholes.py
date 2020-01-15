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
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import lib.black_scholes as bs
from lib.utils import get_duration_since_desc

sns.set(style='darkgrid')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Set this to True if you want to pop up plots of delta vs spot to see how well the approx worked
show_plots = True

# Set this to True if you want to run the MC sim at the end to price the option
price_option = True

# Seed the RNGs so we get consistent results from run to run.
seed = 2
tf.random.set_seed(seed) # Used for parameter initialisation
np.random.seed(seed) # Used to generate Heston paths

# Define the parameters of the stochastic process. Note that the process is defined
# in the real world measure, not the risk neutral one. The process is
#     dS = mu S dt + vol S dz_s
# where the model parameters are mu and vol. mu is the (real world) drift of the asset price S.
S0 = 1 # initial spot price
mu = 0.
vol = 0.2

# Define the parameters of the option we're trying to price. It's a European vanilla option
texp = 0.25 # time to option expiration
K = 1 # option strike price
is_call = True # True, call option; False, put option
phi = 1 if is_call else -1 # Call or put

# Define the parameters of the Monte Carlo simulation we'll run to train the neural network
n_steps = 100 # number of time steps
n_paths = 1000000 # number of MC paths
dt = texp / n_steps
sqrtdt = math.sqrt(dt)

# Set up the neural network that outputs hedge notionals. The inputs are spot
# and the time to expiration;. This network is generic, in the sense that it's not a function of a particular MC path or
# time step or anything like that; it just tells you how to hedge the option in a given state. 
n_layers = 2 # number of hidden layers
n_hidden = 50 # number of hidden layer nodes for both layers
batch_size = 100 # number of MC paths to include in one step of the neural network training
learning_rate = 5e-3 # Adam optimizer initial learning rate
pctile = 70 # percentile for expected shortfall

if n_paths % batch_size != 0:
    raise ValueError('An integer number of batches must fit into the total number of paths')

n_batches = n_paths // batch_size


class Model():
    def __init__(self):
        """Define our NN structure; we use the same nodes in each timestep"""
        # Hidden layers
        self.b_nodes = []
        self.W_nodes = []
        
        for hidden_index in range(n_layers):
            b = tf.Variable(tf.random.truncated_normal([n_hidden]), trainable=True)
            
            if hidden_index == 0:
                W = tf.Variable(tf.random.truncated_normal([2, n_hidden], stddev=0.1), trainable=True)
            else:
                W = tf.Variable(tf.random.truncated_normal([n_hidden, n_hidden], stddev=0.1), trainable= True)
            
            self.b_nodes.append(b)
            self.W_nodes.append(W)
        
        # Output
        # We have one output (notional of spot hedge) is a linear combination of the second
        # hidden layer node values
        self.bo = tf.Variable(tf.random.truncated_normal([1]), trainable=True)
        self.Wo = tf.Variable(tf.random.truncated_normal([n_hidden, 1], stddev=0.1), trainable=True)
        
        # Inputs
        # Our 2 inputs are spot price and time, which are mostly determined during the MC
        # simulation except for the initial spot at time 0
        self.init_spots = tf.Variable(np.zeros(batch_size, dtype=np.float32))
    
    @property
    def trainable_variables(self):
        return self.W_nodes + self.b_nodes + [self.Wo, self.bo]
    
    @tf.function
    def compute_delta(self, x):
        """Returns the output of the neural network at any point in time
        
        The delta size of the position required to hedge the option.
        """
        h = None
        
        for hidden_index in range(n_layers):
            if hidden_index == 0:
                h = tf.nn.relu(tf.matmul(x, self.W_nodes[0]) + self.b_nodes[0])
            else:
                h = tf.nn.relu(tf.matmul(h, self.W_nodes[hidden_index]) + self.b_nodes[hidden_index])
        
        y = tf.matmul(h, self.Wo) + self.bo
        y = tf.identity(y)
        
        return y
    
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
        
        pnl = np.zeros(batch_size).astype(np.float32)
        prev_spot = self.init_spots
        log_spots = np.zeros(batch_size).astype(np.float32)
        
        # Run through the MC sim, generating path values for spots along the way
        for time_index in range(n_steps):
            # Set up the neural network inputs
            t = tf.constant([time_index * dt] * batch_size)
            inputs = tf.stack([prev_spot, t], 1)
            
            # Generate the NN for this time step and get the outputs. By minimising expected
            # shortfall, the output of the network is trained to approximate Black-Scholes delta.
            delta = self.compute_delta(inputs)[:, 0]
            
            rs = np.random.normal(0, sqrtdt, size=batch_size).astype(np.float32)
            log_spots += (mu - vol * vol / 2.) * dt + vol * rs
            
            # Get the spots at the end of the interval and add them to the variable dictionary
            spot = self.init_spots * np.exp(log_spots)
            
            # Calculate the PNL from spot change
            spot_change = tf.subtract(spot, prev_spot)
            inc_pnl = tf.multiply(delta, spot_change)
            pnl = tf.add(pnl, inc_pnl)
            
            # Remember values for the next step
            prev_spot = spot
        
        # Calculate the final payoff
        payoff = tf.maximum(phi * (prev_spot - K), 0)
        pnl = tf.subtract(pnl, payoff) # note we subtract - we sell the option
        
        return pnl
    
    @tf.function
    def compute_loss(self):
        """Use expected shortfall for the appropriate percentile as the loss function.
        
        Note that we do *not* expect this to minimize to zero when we sell an option; with perfect
        hedging, the PNL distribution would be a delta function at -1 * (option premium) because
        we never get paid the upfront premium.
        """
        
        pnl = self.compute_pnl()
        pnl_neg = tf.multiply(pnl, -1)
        n_pct = int((100 - pctile) / 100 * batch_size)
        pnl_past_cutoff = tf.nn.top_k(pnl_neg, n_pct)[0]
        return tf.reduce_mean(pnl_past_cutoff)
    
    @tf.function
    def compute_mean_pnl(self):
        """Mean PNL for debugging purposes"""
        pnl = self.compute_pnl()
        return tf.reduce_mean(pnl)


def train(model):
    """We'll train the network by running an MC simulation, batching up the paths into groups of batch_size paths"""
    # Use the Adam optimizer, which is gradient descent which also evolves
    # the learning rate appropriately (the learning rate passed in is the initial
    # learning rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Remember the start time - we'll log out the total time for training later
    t0 = time.time()
    
    # loop through the `batch_size`-sized subsets of the total paths and train on each
    for batch in range(n_batches):
        # Get a random initial spot so that the training sees a proper range even at t=0.
        # We use the same initial spot price across the batch so that all MC paths are sampled
        # from the same distribution, which is a requirement for our expected shortfall calculation.
        init_spot = S0 * math.exp(-vol * vol * texp / 2. + np.random.normal(0, 2. * vol * math.sqrt(texp)))
        model.init_spots.assign([init_spot] * batch_size)
        
        # Now we've got the inputs set up for the training - run the training step
        optimizer.minimize(model.compute_loss, model.trainable_variables)
        
        # Log some stats as we train
        if batch % batch_size == 0:
            est_mean = -bs.opt_price(is_call, init_spot, K, texp, vol, 0, 0)
            loss_value = model.compute_loss()
            mean_pnl = model.compute_mean_pnl()
            duration = get_duration_since_desc(t0)
            log.info('Batch %04d (%s): loss % .5f, mean % .5f, est mean % .5f, init spot % .5f', batch, duration, loss_value.numpy(), mean_pnl.numpy(), est_mean, init_spot)
    
    duration = get_duration_since_desc(t0)
    log.info('Total training time: %s', duration)


def plot(model):
    """Plot out delta vs spot for a range of calendar times
    
    Calculated against the known closed-form BS delta.
    """
    
    f, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    f.suptitle('Delta vs spot vs time to maturity')
    axes = axes.flatten()
    spot_fact = math.exp(3 * vol * math.sqrt(texp))
    ts = [0, 0.1, 0.2, 0.25]
    
    for t_mid, ax in zip(ts, axes):
        n_spots = 20
        test_spots = np.linspace(S0 / spot_fact, S0 * spot_fact, n_spots).astype(np.float32)
        test_inputs = np.transpose(np.array([test_spots, [t_mid] * n_spots], dtype=np.float32))
        test_deltas = model.compute_delta(test_inputs)
        test_deltas = test_deltas[:, 0]
        log.info('Delta: mean = % .5f, std = % .5f', test_deltas.numpy().mean(), test_deltas.numpy().std())
        est_deltas = [bs.opt_delta(is_call, spot, K, texp - t_mid, vol, 0, 0) for spot in test_spots]
        ax.set_title('Calendar time {:.2f} years'.format(t_mid))
        nn_plot, = ax.plot(test_spots, test_deltas)
        bs_plot, = ax.plot(test_spots, est_deltas)
    
    ax.legend([nn_plot, bs_plot], ['Network', 'Black Scholes'])
    f.text(0.5, 0.04, 'Spot', ha='center')
    f.text(0.04, 0.5, 'Delta', ha='center', rotation='vertical')
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.show()


def calc_opt_price(model):
    """calculate the option price from this optimal hedging strategy. We calculate the expected shortfall; the 
    option price is just that, since adding cash to make it zero is the minimum price we'd need to accept.
    This of course is like an "offer" price because it includes some risk aversion for PNL noise around the 
    mean - but if we've got an accurate hedging strategy then it's not going to be much.
    """
    
    t0 = time.time()
    
    log_spots = np.zeros(n_paths, dtype=np.float32)
    path_pnls = np.zeros(n_paths, dtype=np.float32)
    path_pnls_bs = np.zeros(n_paths, dtype=np.float32)
    prev_spots = np.zeros(n_paths, dtype=np.float32) + S0
    
    inputs = np.transpose(np.array([prev_spots, [0] * n_paths], dtype=np.float32))
    deltas = model.compute_delta(inputs)
    deltas = deltas[:, 0]
    
    deltas_bs = [bs.opt_delta(is_call, float(spot), K, texp, vol, 0, 0) for spot in prev_spots]
    
    # Run through the MC sim, generating path values for spots along the way. This is just like a regular MC
    # sim to price a derivative - except that the price is *not* the expected value - it's the loss function
    # value. That handles both the conversion from real world to "risk neutral" and unhedgeable risk due to
    # eg discrete hedging (which is the only unhedgeable risk in this example, but there could be anything generally).
    for time_index in range(n_steps):
        rs = np.random.normal(0, sqrtdt, n_paths)
        log_spots += (mu - vol * vol / 2.) * dt + vol * rs
        
        # Get the PNLs of the hedges over the interval
        
        t = (time_index + 1) * dt # end of interval
        spots = S0 * np.exp(log_spots)
        
        path_pnls += deltas * (spots - prev_spots)
        path_pnls_bs += deltas_bs * (spots - prev_spots)
        
        # Get the next step hedge notionals
        inputs = tf.constant(np.transpose(np.array([spots, [t] * n_paths], dtype=np.float32)))
        deltas = model.compute_delta(inputs)
        deltas = deltas[:, 0]
        
        # Also get the deltas used Black-Scholes
        deltas_bs = [bs.opt_delta(is_call, float(spot), K, texp - t, vol, 0, 0) for spot in spots]
        
        # Remember stuff for the next time step
        prev_spots = spots

        log.info('%.4f years - delta: mean % .5f, std % .5f; spot: mean % .5f, std % .5f', time_index * dt, deltas.numpy().mean(), deltas.numpy().std(), spots.mean(), spots.std())
    
    # Compute the payoff some metrics
    payoff = np.maximum(phi * (spots - K), 0)
    path_pnls -= payoff
    path_pnls_bs -= payoff
    
    n_pct = int((100 - pctile) / 100 * n_paths)
    path_pnls = np.sort(path_pnls)
    price_dh = -path_pnls[:n_pct].mean()
    path_pnls_bs = np.sort(path_pnls_bs)
    price_bs_es = -path_pnls_bs[:n_pct].mean()
    log.info('Deep hedging price = % .5f', price_dh)
    log.info('Hedging price BS   = % .5f', price_bs_es)
    log.info('BS price           = % .5f', bs.opt_price(is_call, S0, K, texp, vol, 0, 0))
    log.info('Mean payoff        = % .5f', payoff.mean())
    log.info('Mean PNL           = % .5f', path_pnls.mean())
    log.info('Std dev PNL        = % .5f', path_pnls.std())
    
    duration = get_duration_since_desc(t0)
    log.info('Simulation time: %s', duration)


def main():
    """Run the model"""
    
    model = Model()
    train(model)
    
    if show_plots:
        plot(model)
    
    if price_option:
        calc_opt_price(model)


if __name__ == '__main__':
    main()
