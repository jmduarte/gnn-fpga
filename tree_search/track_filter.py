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
from torch.autograd import Variable

# Local imports
import torchutils

# Scale (phi,z,r) down to ~unit max
coord_scale = np.array([0.1*np.pi, 1000., 1000.])

def remove_duplicate_hits(hits):
    """Averages together all duplicate (same particle) hits on layers"""
    return hits.groupby(['evtid', 'barcode', 'layer'], as_index=False).mean()

def remove_duplicate_hits_2(hits):
    """
    Removes duplicate (same particle) hits on layers by keeping only the
    ones with smallest cylindrical radius.
    """
    return hits.loc[
        hits.groupby(['evtid', 'barcode', 'layer'], as_index=False)
        .r.idxmin()
    ]

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
    return remove_duplicate_hits_2(
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

class Cholesky(torch.autograd.Function):
    """
    Cholesky decomposition with gradient. Taken from
    https://github.com/t-vi/pytorch-tvmisc/blob/master/misc/gaussian_process_regression_basic.ipynb
    """
    @staticmethod
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l

    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}
        # Ideally, this should use some form of solve triangular instead of inverse...
        linv =  l.inverse()

        inner = (torch.tril(torch.mm(l.t(), grad_output)) *
                 torch.tril(1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag())))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class HitGausPredictor(nn.Module):
    """
    A PyTorch module for particle track state estimation and hit prediction.

    This module is an RNN which takes a sequence of hits and produces a
    Gaussian shaped prediction for the location of the next hit.
    """

    def __init__(self, hidden_dim=5):
        super(HitGausPredictor, self).__init__()
        input_dim = 3
        output_dim = 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        out_size = input_dim * (input_dim + 3) / 2
        self.fc = nn.Linear(hidden_dim, out_size)

    def forward(self, x):
        """Might want to accept also the radius of the target layer."""
        input_size = x.size()

        # Initialize the LSTM hidden state
        h = (torchutils.torch_zeros(input_size[0], self.lstm.hidden_size),
             torchutils.torch_zeros(input_size[0], self.lstm.hidden_size))
        # Apply the LSTM module
        x, h = self.lstm(x, h)
        # Squash layer axis into batch axis
        x = x.contiguous().view(-1, x.size(-1))
        # Apply linear layer
        output = self.fc(x)

        # Extract and transform the gaussian parameters
        means = output[:, :2]
        variances = output[:, 2:4].exp()
        correlations = output[:, 4].tanh()

        # Construct the covariance matrix
        covs = torch.bmm(variances[:, :, None], variances[:, None, :]).sqrt()
        covs[:, 0, 1] = covs[:, 0, 1].clone() * correlations
        covs[:, 1, 0] = covs[:, 1, 0].clone() * correlations

        # Expand the layer axis again, just for consistency/interpretability
        means = means.contiguous().view(input_size[0], input_size[1], 2)
        covs = covs.contiguous().view(input_size[0], input_size[1], 2, 2)
        return means, covs

def gaus_llh_loss(outputs, targets):
    """Custom gaussian log-likelihood loss function"""
    means, covs = outputs
    # Flatten layer axis into batch axis to use batch matrix operations
    means = means.contiguous().view(means.size(0)*means.size(1), means.size(2))
    covs = covs.contiguous().view(covs.size(0)*covs.size(1),
                                  covs.size(2), covs.size(3))
    targets = targets.contiguous().view(targets.size(0)*targets.size(1),
                                        targets.size(2))
    # Calculate the inverses of the covariance matrices
    inv_covs = torch.stack([cov.inverse() for cov in covs])
    # Calculate the residual error
    # TODO: need to fix for phi discontinuity!!
    res = targets - means
    # Calculate the residual error term
    res_right = torch.bmm(inv_covs, res.unsqueeze(-1)).squeeze(-1)
    res_term = torch.bmm(res[:,None,:], res_right[:,:,None]).squeeze()
    # For the determinant term, we first have to compute the cholesky roots
    diag_chols = torch.stack([Cholesky.apply(cov).diag() for cov in covs])
    log_det = diag_chols.log().sum(1) * 2
    gllh_loss = (res_term + log_det).sum()
    return gllh_loss
