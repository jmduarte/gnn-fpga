"""
Module containing all code specific to the track filter
(aka hit predictor) models.
That includes the models themselves, but also the feature extraction code.
"""

# Data libraries
import numpy as np
import pandas as pd

# Deep learning
import torch
import torch.nn as nn

# Local imports
import torchutils

# Scale (phi,z,r) down to ~unit max
coord_scale = np.array([np.pi, 1000., 1000.])

def remove_duplicate_hits(hits):
    """Averages together all duplicate (same particle) hits on layers"""
    return hits.groupby(['evtid', 'barcode', 'layer'], as_index=False).mean()

def select_hits(hits):
    # Select all barrel hits
    vids = [8, 13, 17]
    barrel_hits = hits[np.logical_or.reduce([hits.volid == v for v in vids])]
    # Re-enumerate the volume and layer numbers for convenience
    volume = pd.Series(-1, index=barrel_hits.index, dtype=np.int8)
    vid_groups = barrel_hits.groupby('volid')
    for i, v in enumerate(vids):
        volume[vid_groups.get_group(v).index] = i
    # This assumes 4 layers per volume (except last volume)
    layer = (barrel_hits.layid / 2 - 1 + volume * 4).astype(np.int8)
    return (barrel_hits[['evtid', 'barcode', 'r', 'phi', 'z']]
            .assign(volume=volume, layer=layer))

def select_signal_hits(hits):
    """Select signal hits from tracks that hit all barrel layers"""
    return remove_duplicate_hits(
            hits.groupby(['evtid', 'barcode'])
            .filter(lambda x: len(x) >= 10 and x.layer.unique().size == 10))

class HitPredictor(nn.Module):
    """RNN which predicts sequence of next-hits from input sequence"""

    def __init__(self, input_dim=3, hidden_dim=5, output_dim=2,
                 n_lstm_layers=1):
        super(HitPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        input_size = x.size()
        # Initialize the lstm hidden state
        torch_zeros = torchutils.torch_zeros
        func_args = (self.lstm.num_layers, input_size[0], self.lstm.hidden_size)
        h = (torch_zeros(*func_args), torch_zeros(*func_args))
        x, h = self.lstm(x, h)
        # Flatten layer axis into batch axis so FC applies
        # independently across layers.
        x = (self.fc(x.contiguous().view(-1, x.size(-1)))
             .view(input_size[0], input_size[1], -1))
        return x

class HitGausPredictor(nn.Module):
    """TODO: move my model here"""
    pass
