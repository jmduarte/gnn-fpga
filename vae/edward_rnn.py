import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers
from keras.engine import InputSpec
import keras.backend as K
import edward as ed
from edward.models import Bernoulli, Normal, PointMass

hits_row_names = ["event_id", "track_id", "i_r", "i_phi", "x", "y"]
max_layer_hits = 25
n_seed_layers = 3
n_layers = 9
n_target_layers = n_layers - n_seed_layers

class StochasticRNN(object):
    """A RNN layer with a stochastic hidden state."""

    def __init__(self, in_size, out_size, name):
        # dimension of input
        self.in_size = in_size
        # dimension of hidden state
        self.out_size = out_size

        self.name = name
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.zero_initializer = tf.constant_initializer(0.0)

        # This is the usual set of RNN weights
        self.Wx_mu = self.make_weights((in_size, out_size), 'Wx_mu')
        self.Wx_sigma = self.make_weights((in_size, out_size), 'Wx_sigma')
        self.Wz_mu = self.make_weights((out_size, out_size), 'Wz_mu')
        self.Wz_sigma = self.make_weights((out_size, out_size), 'Wzp_sigma')
        self.b_mu = self.make_weights((out_size,), 'b_mu', initialize=False)
        self.b_sigma = self.make_weights((out_size,), 'b_sigma', initialize=False)
        self.W_mu = self.make_weights((out_size, out_size), 'W_mu')
        self.W_sigma = self.make_weights((out_size, out_size), 'W_sigma')

    def make_weights(self, shape, name, initialize=True):
        """
        shape: tuple
        name: string
        initialize: use glorot initialization for weights
        """
        with tf.variable_scope(self.name):
            if initialize:
                initializer = self.initializer
            else:
                initializer = self.zero_initializer
            v = tf.get_variable(name, shape, initializer=initializer)
        return v

    def z_recurrence(self, prev, x):
        """
        prev: previous hidden state
        x: current input
        """
        ### This should return the random variable Z 
        ### but I was having issues getting this to work.
        ### Needs debugging.
        ### For now I generate the mean and std of the
        ### random variable and return those.  
        mu_prev, sigma_prev = prev
        z_prev = Normal(mu=mu_prev, sigma=sigma_prev)
        pre_mu = tf.tanh(K.dot(z_prev, self.Wz_mu) + 
                K.dot(x, self.Wx_mu) + self.b_mu)
        pre_sigma = tf.tanh(K.dot(z_prev, self.Wz_sigma) + 
                K.dot(x, self.Wx_sigma) + self.b_sigma)
        z_mu = K.dot(pre_mu, self.W_mu)
        z_sigma = tf.nn.softplus(K.dot(pre_sigma, self.W_sigma))
        return [z_mu, z_sigma]

    def run(self, inputs):
        # tf.scan() needs time dimension to be dim 0
        initial = K.zeros_like(inputs) # batch, tsteps, hidden
        initial = K.sum(initial, axis=(1,2)) # samples,
        initial = K.expand_dims(initial) # samples, 1
        initial = K.tile(initial, [1, self.out_size]) # samples, out_size
        initial = [initial for _ in range(2)]
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = tf.scan(self.z_recurrence, inputs, 
                initializer=initial)
        outputs = [tf.transpose(out, [1, 0, 2]) for out in outputs]
        return outputs


class EdwardLSTM(object):
    """From Fraccaro et al, 
       `Sequential Neural Models with Stochastic Layers.'
       Attributes:
       -hidden_size: number of LSTM hidden units
       -in_size: input hit positions in each layer will be
            transformed into a fixed vector of this length
    """
    def __init__(self, hidden_size, in_size):
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.build()

    def build(self):
        ### decoder network

        # the seed hits are first encoded using an LSTM.
        # the encoded seed is used as the initial hidden state below.
        self.seed_input = layers.Input(shape=(n_seed_layers,1))
        seeds = layers.Masking(mask_value=-1)(self.seed_input)
        seeds_1 = layers.LSTM(self.hidden_size)(seeds)
        seeds_1 = layers.Activation('tanh')(seeds_1)
        seeds_2 = layers.LSTM(self.hidden_size)(seeds)
        seeds_2 = layers.Activation('tanh')(seeds_2)

        # run an LSTM on each layer's list of input hits 
        # to transform them to a fixed representation.
        self.hit_input = layers.Input((n_target_layers, 
            max_layer_hits, 1))
        hits = layers.Masking(mask_value=-1)(self.hit_input)
        hits = layers.TimeDistributed(
                layers.LSTM(self.in_size))(hits)
        hits = layers.Activation('tanh')(hits)

        # this LSTM represents the deterministic part of the decoder
        forward_lstm = layers.LSTM(self.hidden_size,
                return_sequences=True)(hits, initial_state=[seeds_1, seeds_2])

        # this RNN represents the stochastic part of the decoder
        z_rnn = StochasticRNN(self.hidden_size, self.hidden_size, 'p')
        z_mu, z_sigma = z_rnn.run(forward_lstm)
        z = Normal(mu=z_mu, sigma=z_sigma)

        # this MLP decodes deterministic and stochastic latent states into data
        decoder_output = tf.concat([forward_lstm, z], axis=2)
        decoder_output = layers.TimeDistributed(layers.Dense(self.hidden_size)
            )(decoder_output)
        #Making the output a random variable causes me to experience this:
        #https://github.com/blei-lab/edward/issues/525
        #x_mu = layers.TimeDistributed(layers.Dense(1))(decoder_output)
        #x_mu = tf.reshape(x_mu, [-1, n_target_layers])
        #x_sigma = layers.TimeDistributed(layers.Dense(1))(decoder_output)
        #x_sigma = tf.nn.softplus(tf.reshape(
        #    x_sigma, [-1, n_target_layers]))
        #self.x = Normal(mu=x_mu, sigma=x_sigma)
        x = layers.TimeDistributed(layers.Dense(1))(decoder_output)
        self.x = tf.reshape(x, [-1, n_target_layers])

        ### encoder network
        
        self.x_obs = tf.placeholder(tf.float32, [None, n_target_layers])
        self.scale = tf.placeholder(tf.float32, [None, n_target_layers])
        backward_lstm = layers.LSTM(self.hidden_size, return_sequences=True,
                go_backwards=True)(forward_lstm, initial_state=[seeds_1, seeds_2])
        z_q_rnn = StochasticRNN(self.hidden_size, self.hidden_size, 'q')
        z_backward_mu, z_backward_sigma = z_q_rnn.run(backward_lstm)
        self.zq_variables = [z_backward_mu, z_backward_sigma]
        z_q = Normal(mu=z_mu+z_backward_mu,
                sigma=z_backward_sigma)

        ### inference procedure

        self.inference = ed.KLqp({z: z_q}, data={self.x: self.x_obs})
        self.optimizer = tf.train.AdamOptimizer()
        self.inference.initialize(optimizer=self.optimizer, 
                scale={self.x:self.scale}, logdir=".")

    def train(self, generator, n_batches, batch_size, print_every=10):
        self.sess = ed.get_session()
        init = tf.global_variables_initializer()
        init.run()
        avg_loss = 0
        for i_batch in range(n_batches):
            seeds, hits, targets, weights = generator.next()
            info = self.inference.update(feed_dict={
                    self.x_obs:targets,
                    self.seed_input:seeds, 
                    self.hit_input:hits,
                    self.scale:weights,
                    })
            avg_loss += info['loss']
            if not i_batch % print_every and i_batch > 0:
                avg_loss /= (print_every * batch_size)
                print "Batch {} of {}: loss = {:0.3f}".format(
                        i_batch, n_batches, avg_loss)
                avg_loss = 0
                self.display_progress(seeds, hits, targets, weights)

    def display_progress(self, seeds, hits, targets, weights):
        #preds = self.x.value().eval() #once x is a random var
        feed = {
                self.seed_input:seeds,
                self.hit_input:hits,
                }
        preds = self.sess.run(self.x, feed_dict=feed)
        print "Target:"
        print targets[0]
        print "Prediction:"
        print preds[0]

def get_phi(x, y):
    """Gives phi in the range [0, 2*pi)"""
    return np.arctan2(y, x) + np.pi

def get_train_event(evt, track_id):
    """
    evt: pandas dataframe holding hit information for one event
    track_id: number of track to reconstruct
    
    Returns the seed hits of the target track (first three layers)
    and the phi coordinates of all hits in subsequent layers
    as well as the weights used for masking missing hits.
    """
    seed_hits = -np.ones((n_seed_layers, 1))
    layer_hits = -np.ones((n_target_layers, max_layer_hits, 1))
    layer_targets = -np.ones(n_target_layers)
    layer_weights = np.zeros(n_target_layers)
    # This list keeps track of indices within layer_hits
    pos = [0 for _ in range(n_target_layers)] 
    try:
        for hit in evt.itertuples():
            itrack = hit[hits_row_names.index('track_id')]
            ir = hit[hits_row_names.index('i_r')]

            x = hit[hits_row_names.index('x')]
            y = hit[hits_row_names.index('y')]
            phi = get_phi(x, y)

            if itrack==track_id:
                if ir < n_seed_layers:
                    seed_hits[ir,0] = phi
                else:
                    layer_targets[ir-n_seed_layers] = phi
                    layer_weights[ir-n_seed_layers] = 1.0

            if ir >= n_seed_layers:
                ind = ir-n_seed_layers
                if pos[ind] < max_layer_hits:
                    layer_hits[ind, pos[ind], 0] = phi
                    pos[ind] += 1
    except AttributeError:
        # This occurs if the event has only one hit (rare), in which case evt
        # is a Series, not a DataFrame.  Deal with this separately.
        print "Encountered event with only one hit:",evt
        
    return seed_hits, layer_hits, layer_targets, layer_weights

def gen_single_hits(hit_files):
    """
    hit_files: list of paths to input data files.
    Yields tuples (seed hits, layer hits, layer targets, weights)
    """
    cur_file = 0
    num_files = len(hit_files)
    while True:
        df = pd.read_csv(hit_files[cur_file], header=None, 
                names=hits_row_names, index_col=hits_row_names[0])
        event_nums = sorted(df.index.unique())
        for event_num in event_nums:
            # for each event, pick a random track
            evt = df.loc[event_num]
            track_id = np.random.randint(evt['track_id'].max()+1)
            yield get_train_event(evt, track_id)
        cur_file += 1
        if cur_file >= num_files:
            cur_file = 0

def generate_data(batch_size, hit_files):
    gen_hits = gen_single_hits(hit_files)
    while True:
        batch_seeds = np.zeros((batch_size, n_seed_layers, 1))
        batch_layer_hits = np.zeros((batch_size, n_target_layers, 
                max_layer_hits, 1))
        batch_targets = np.zeros((batch_size, n_target_layers))
        batch_weights = np.zeros((batch_size, n_target_layers))
        for n in range(batch_size):
            seed, layer_hits, target_hits, target_weights = gen_hits.next()
            batch_seeds[n] = seed
            batch_layer_hits[n] = layer_hits
            batch_targets[n] = target_hits
            batch_weights[n] = target_weights
        yield batch_seeds, batch_layer_hits, batch_targets, batch_weights


if __name__ == '__main__':
    hit_files = glob.glob('hits_*.csv')
    batch = 256
    nevents = 448000
    epochs = 1
    gen = generate_data(batch_size=batch, hit_files=hit_files)
    model = EdwardLSTM(170, 170)
    model.train(gen, epochs * nevents / batch, batch)
