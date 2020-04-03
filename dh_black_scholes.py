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
import seaborn as sns
import tensorflow as tf

import models.european_option.analytics as analytics
from lib.utils import get_duration_desc

sns.set(style='darkgrid')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Control toggles
do_train = True # Actually train the model
show_delta_plot = True # Pop up plot of delta vs spot
show_pnl_plot = True # Run MC sim to compute PnL

# Seed the RNGs so we get consistent results from run to run.
seed = 2
tf.random.set_seed(seed) # Used for parameter initialisation
np.random.seed(seed) # Used to generate Heston paths

# Define the parameters of the stochastic process. Note that the process is defined
# in the real world measure, not the risk neutral one. The process is
#     dS = mu S dt + vol S dz_s
# where the model parameters are mu and vol. mu is the (real world) drift of the asset price S.
S0 = 1.0 # initial spot price
mu = 0.0 # Expected upward spot drift, in years
vol = 0.2 # Volatility

# Define the parameters of the option we're trying to price. It's a European vanilla option
texp = 0.25 # time to option expiration
K = 1 # option strike price
is_call = True # True: call option; False: put option
is_buy = False # True: buying a call/put; False: selling a call/put
phi = 1 if is_call else -1 # Call or put
psi = 1 if is_buy else -1 # Buy or sell

# Define the parameters of the Monte Carlo simulation we'll run to train the neural network
n_steps = 100 # number of time steps
n_paths = 1_000_000 # number of MC paths
dt = texp / n_steps
sqrtdt = dt ** 0.5

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
                W = tf.Variable(tf.random.truncated_normal([n_hidden, n_hidden], stddev=0.1), trainable=True)
            
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
        self.init_spot = tf.Variable(S0)
    
    @property
    def trainable_variables(self):
        return self.W_nodes + self.b_nodes + [self.Wo, self.bo]
    
    @tf.function
    def compute_hedge_delta(self, x):
        """Returns the output of the neural network at any point in time
        
        The delta size of the position required to hedge the option.
        """
        h = x
        
        for hidden_index in range(n_layers):
            h = tf.nn.relu(tf.matmul(h, self.W_nodes[hidden_index]) + self.b_nodes[hidden_index])
        
        y = tf.matmul(h, self.Wo) + self.bo
        
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
        expected shortfall of the integrated PNLs for each path in a batch. Make sure
        we're short the option so that (absent hedging) there's a -ve PNL.
        """
        
        pnl = tf.zeros(batch_size, dtype=tf.float32)
        spot = tf.zeros(batch_size, dtype=tf.float32) + self.init_spot
        log_spot = tf.zeros(batch_size, dtype=tf.float32)
        
        # Run through the MC sim, generating path values for spots along the way
        for time_index in tf.range(n_steps, dtype=tf.float32):
            # Retrieve the neural network output, treating it as the delta hedge notional
            # at the start of the timestep. By minimising expected shortfall, the output
            # of the network is trained to approximate Black-Scholes delta.
            t = time_index * dt
            input_time = tf.fill([batch_size], t)
            nn_input = tf.stack([spot, input_time], 1)
            delta = self.compute_hedge_delta(nn_input)[:, 0]
            
            # Get the spots at the end of the interval
            rs = tf.random.normal([batch_size], 0, sqrtdt)
            log_spot += (mu - vol * vol / 2.) * dt + vol * rs
            new_spot = self.init_spot * tf.math.exp(log_spot)
            
            # Calculate the PNL from spot change and dynamically delta hedge
            spot_change = new_spot - spot
            pnl += delta * spot_change
            
            # Remember values for the next step
            spot = new_spot
        
        # Calculate the final payoff
        payoff = psi * tf.maximum(phi * (spot - K), 0)
        pnl += payoff # Note we sell the option here
        
        return pnl
    
    @tf.function
    def compute_loss(self):
        """Use expected shortfall for the appropriate percentile as the loss function.
        
        Note that we do *not* expect this to minimize to zero when we sell an option; with perfect
        hedging, the PNL distribution would be a delta function at -1 * (option premium) because
        we never get paid the upfront premium.
        """
        
        pnl = self.compute_pnl()
        n_pct = int((100 - pctile) / 100 * batch_size)
        pnl_past_cutoff = tf.nn.top_k(-pnl, n_pct)[0]
        return tf.reduce_mean(pnl_past_cutoff) / self.init_spot
    
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
    losses = []
    
    # loop through the `batch_size`-sized subsets of the total paths and train on each
    for batch in range(n_batches):
        # Get a random initial spot so that the training sees a proper range even at t=0.
        # We use the same initial spot price across the batch so that all MC paths are sampled
        # from the same distribution, which is a requirement for our expected shortfall calculation.
        init_spot = S0 * np.exp(-vol * vol * texp / 2. + np.random.normal(0, 2. * vol * np.sqrt(texp)))
        model.init_spot.assign(init_spot)
        
        # Now we've got the inputs set up for the training - run the training step
        optimizer.minimize(model.compute_loss, model.trainable_variables)
        losses.append(model.compute_loss())
        
        # Log some stats as we train
        if batch % batch_size == 0:
            est_mean = psi * analytics.opt_price(is_call, init_spot, K, texp, vol, 0, 0)
            loss_value = model.compute_loss()
            mean_pnl = model.compute_mean_pnl()
            duration = get_duration_desc(t0)
            log.info('Batch %04d (%s): loss % .5f, mean % .5f, est mean % .5f, init spot % .5f', batch, duration, loss_value.numpy(), mean_pnl.numpy(), est_mean, init_spot)
    
    duration = get_duration_desc(t0)
    log.info('Total training time: %s', duration)
    plot_loss(losses)


def calc_expected_shortfall(pnls):
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
    
    n_pct = int((100 - pctile) / 100 * n_paths)
    pnls = np.sort(pnls)
    price = pnls[:n_pct].mean()
    
    return price


def calc_pnls(model, n_paths=100_000):
    """Calculate the pnls from this optimal hedging strategy.
    
    Parameters
    ----------
    model : :obj:`Model`
        Trained model.
    n_paths : int
        Number of paths to simulate.
    
    Returns
    -------
    tuple of :obj:`numpy.array`
        (unhedged pnl, Black-Scholes hedged pnl, neural network hedged pnl)
    """
    
    t0 = time.time()
    
    log_spot = np.zeros(n_paths, dtype=np.float32)
    nn_pnls = np.zeros(n_paths, dtype=np.float32)
    bs_pnls = np.zeros(n_paths, dtype=np.float32)
    spot = np.zeros(n_paths, dtype=np.float32) + S0
    
    # Run through the MC sim, generating path values for spots along the way. This is just like a regular MC
    # sim to price a derivative - except that the price is *not* the expected value - it's the loss function
    # value. That handles both the conversion from real world to "risk neutral" and unhedgeable risk due to
    # eg discrete hedging (which is the only unhedgeable risk in this example, but there could be anything generally).
    for time_index in range(n_steps):
        # Get the hedge notionals at the start of the timestep
        t = time_index * dt # Start of interval
        input_time = tf.constant([t] * n_paths)
        nn_input = tf.stack([spot, input_time], 1)
        nn_delta = model.compute_hedge_delta(nn_input)[:, 0].numpy()
        
        # Also get the deltas used Black-Scholes
        bs_delta = -psi * analytics.opt_delta(is_call, spot, K, texp - t, vol, 0, 0)
        
        # Advance spot to the end of the timestep
        rs = np.random.normal(0, sqrtdt, n_paths)
        log_spot += (mu - vol * vol / 2.) * dt + vol * rs
        new_spot = S0 * np.exp(log_spot)
        
        # Get the PNLs of the hedges over the interval
        spot_change = new_spot - spot
        nn_pnls += nn_delta * spot_change
        bs_pnls += bs_delta * spot_change
        
        # Remember stuff for the next time step
        spot = new_spot
        log.info('%.4f years - delta: mean % .5f, std % .5f; spot: mean % .5f, std % .5f', t, nn_delta.mean(), nn_delta.std(), spot.mean(), spot.std())
    
    # Compute the payoff and some metrics
    payoff = psi * np.maximum(phi * (spot - K), 0)
    nn_pnls += payoff
    bs_pnls += payoff
    
    # Report stats
    nn_price = psi * calc_expected_shortfall(nn_pnls)
    bs_price_es = psi * calc_expected_shortfall(bs_pnls)
    bs_price = -psi * analytics.opt_price(is_call, S0, K, texp, vol, 0, 0)
    duration = get_duration_desc(t0)
    
    log.info('Deep hedging price = % .5f', nn_price)
    log.info('BS hedging price   = % .5f', bs_price_es)
    log.info('BS price           = % .5f', bs_price)
    log.info('Mean payoff        = % .5f', payoff.mean())
    log.info('DH mean PNL        = % .5f', nn_pnls.mean())
    log.info('DH std dev PNL     = % .5f', nn_pnls.std())
    log.info('Simulation time: %s', duration)
    
    # Unhedged, bs hedged, nn hedged
    return payoff, bs_pnls, nn_pnls


def plot_loss(losses, window1=50, window2=500):
    smoothed1 = np.convolve(losses, np.ones((window1,)) / window1, mode='valid')
    smoothed2 = np.convolve(losses, np.ones((window2,)) / window2, mode='valid')
    plt.plot(losses)
    plt.plot(smoothed1)
    plt.plot(smoothed2)
    plt.title('Loss over time')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(labels=['Loss', f'Mean of {window1}', f'Mean of {window2}'])
    plt.tight_layout()
    plt.show()


def plot_deltas(model):
    """Plot out delta vs spot for a range of calendar times
    
    Calculated against the known closed-form BS delta.
    
    Parameters
    ----------
    model : `Model`
        Trained model.
    """
    
    f, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    f.suptitle('Option delta hedge vs spot vs time to maturity')
    axes = axes.flatten()
    spot_fact = np.exp(3 * vol * texp ** 0.5)
    ts = [0., 0.1, 0.2, 0.25]
    n_spots = 50
    
    for t, ax in zip(ts, axes):
        # Prepare test inputs
        test_spot = np.linspace(S0 / spot_fact, S0 * spot_fact, n_spots).astype(np.float32)
        test_input = np.transpose(np.array([test_spot, [t] * n_spots], dtype=np.float32))
        
        # Compute neural network delta
        test_delta = model.compute_hedge_delta(test_input)[:, 0].numpy()
        log.info('Delta: mean = % .5f, std = % .5f', test_delta.mean(), test_delta.std())
        
        # Compute Black Scholes delta
        # The hedge will have the opposite sign as the option we are hedging,
        # ie the hedge of a long call is a short call, so we flip psi.
        est_deltas = -psi * analytics.opt_delta(is_call, test_spot, K, texp - t, vol, 0, 0)
        
        # Add a subsplot
        ax.set_title('Calendar time {:.2f} years'.format(t))
        nn_plot, = ax.plot(test_spot, test_delta)
        bs_plot, = ax.plot(test_spot, est_deltas)
    
    ax.legend([nn_plot, bs_plot], ['Network', 'Black Scholes'])
    f.text(0.5, 0.04, 'Spot', ha='center')
    f.text(0.04, 0.5, 'Delta', ha='center', rotation='vertical')
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.show()


def plot_pnls(pnls, labels=None, trim_tails=0):
    """Plot histogram comparing pnls
    
    pnls : list of :obj:`numpy.array`
        Pnls to plot
    labels : list of str
        Labels to add to the legend, corresponding to pnls
    trim_tails : int
        Percentile to trim from each tail when plotting
    """
    
    hist_range = (np.percentile(pnls, trim_tails), np.percentile(pnls, 100 - trim_tails))
    plt.hist(pnls, range=hist_range, bins=100, edgecolor='none')
    plt.title('Post-hedge PNL histogram')
    plt.xlabel('PNL')
    plt.ylabel('Frequency')
    
    if labels is not None:
        plt.legend(labels)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run the model"""
    
    log.info('Hedging a %s %s', 'long' if is_buy else 'short', 'call' if is_call else 'put')
    
    model = Model()
    
    if do_train:
        train(model)
    
    if show_delta_plot:
        plot_deltas(model)
    
    if show_pnl_plot:
        pnls = calc_pnls(model)
        plot_pnls(pnls, labels=['Unhedged', 'Black-Scholes', 'Deep Hedging'], trim_tails=5)


if __name__ == '__main__':
    main()
