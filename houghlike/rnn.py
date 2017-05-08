import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from keras import models, layers

hits_row_names = ["event_id", "track_id", "i_r", "i_phi", "x", "y"]
max_layer_hits = 25
n_seed_layers = 3
n_layers = 9
n_target_layers = n_layers - n_seed_layers

class RampLSTM(object):
    """
       -hidden_size: number of LSTM hidden units
       -in_size: input hit positions in each layer will be
            transformed into a fixed vector of this length
    """
    def __init__(self, hidden_size, in_size, batch):
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.batch = batch
        self.build()

    def build(self):
        # the seed hits are first encoded using an LSTM.
        # the encoded seed is used as the initial cell state below.
        self.seed_input = layers.Input(shape=(n_seed_layers,1))
        seeds = layers.Masking(mask_value=-1)(self.seed_input)
        seeds_forward = layers.LSTM(self.hidden_size)(seeds)
        seeds_forward = layers.Activation('tanh')(seeds_forward)
        seeds_backward = layers.LSTM(self.hidden_size)(seeds)
        seeds_backward = layers.Activation('tanh')(seeds_backward)

#        seeds_for_hits = layers.LSTM(self.in_size)(seeds)
#        seeds_for_hits = layers.Activation('tanh')(seeds_for_hits)

        # the initial hidden state is 0 for each LSTM
        zeros = layers.Lambda(lambda x: K.zeros_like(x))(seeds_forward)
#        zeros_for_hits = layers.Lambda(lambda x: K.zeros_like(x))(seeds_for_hits)

        # run an LSTM on each layer's list of input hits 
        # to transform them to a fixed representation.
        self.hit_input = layers.Input((n_target_layers, 
            max_layer_hits, 1))
        hits = layers.TimeDistributed(
                layers.Masking(mask_value=-1))(self.hit_input)
        hits = layers.TimeDistributed(
                layers.LSTM(self.in_size))(hits)
        hits = layers.Activation('tanh')(hits)

        forward_lstm = layers.LSTM(self.hidden_size,
                return_sequences=True)(hits, 
                        initial_state=[zeros, seeds_forward])
        backward_lstm = layers.LSTM(self.hidden_size, return_sequences=True,
                go_backwards=True)(forward_lstm, 
                        initial_state=[zeros, seeds_backward])

        output = layers.TimeDistributed(layers.Dense(self.hidden_size)
            )(backward_lstm)
        output = layers.TimeDistributed(layers.Dense(1))(output)

        self.model = models.Model(inputs=[self.seed_input, self.hit_input],
                outputs=[output])
        self.model.compile(loss='mse', optimizer='adam', 
                sample_weight_mode='temporal')

    def train(self, generator, epoch_size, epochs):
        self.model.fit_generator(generator, epoch_size, epochs)


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
    layer_targets = -np.ones((n_target_layers, 1))
    layer_weights = np.zeros((n_target_layers,))
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
                    layer_targets[ir-n_seed_layers,0] = phi
                    layer_weights[ir-n_seed_layers] = 1.0

            if ir >= n_seed_layers:
                ind = ir-n_seed_layers
                if pos[ind] < max_layer_hits:
                    layer_hits[ind, pos[ind], 0] = phi
                    pos[ind] += 1
    except AttributeError:
        # This occurs if the event has only one hit (rare), in which case evt
        # is a Series, not a DataFrame.  Deal with this separately.
        #print "Encountered event with only one hit:",evt
        pass
        
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
        batch_targets = np.zeros((batch_size, n_target_layers, 1))
        batch_weights = np.zeros((batch_size, n_target_layers,))
        for n in range(batch_size):
            seed, layer_hits, target_hits, target_weights = gen_hits.next()
            batch_seeds[n] = seed
            batch_layer_hits[n] = layer_hits
            batch_targets[n] = target_hits
            batch_weights[n] = target_weights
        yield [batch_seeds, batch_layer_hits], batch_targets, batch_weights


if __name__ == '__main__':
    hit_files = glob.glob('hits_*.csv')
    batch = 256
    epoch_size = 448000/batch
    epochs = 20
    gen = generate_data(batch_size=batch, hit_files=hit_files)
    model = RampLSTM(1000, 100, batch=batch)

    model.train(gen, epoch_size, epochs)
    model.model.save("rnn.h5")
