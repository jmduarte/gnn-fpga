#!/usr/bin/env python

from __future__ import print_function

import os
import logging
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# Local imports
import torchutils
from estimator import Estimator
from track_filter import HitPredictor

def parse_args():
    parser = argparse.ArgumentParser('trainTrackFilter.py')
    add_arg = parser.add_argument
    add_arg('--input-dir', default='/global/cscratch1/sd/sfarrell/heptrkx/RNNFilter')
    add_arg('--output-dir')
    add_arg('--n-train', type=int, help='Maximum number of training samples')
    add_arg('--n-valid', type=int, help='Maximum number of validation samples')
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--batch-size', type=int, default=32)
    add_arg('--hidden-dim', type=int, default=20)
    add_arg('--cuda', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

class TrackFilterer(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=5, output_dim=2, n_lstm_layers=1):
        super(TrackFilterer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        input_size = x.size()
        # Initialize the lstm hidden state
        torch_zeros = torchutils.torch_zeros
        h = (torch_zeros(self.lstm.num_layers, input_size[0], self.lstm.hidden_size),
             torch_zeros(self.lstm.num_layers, input_size[0], self.lstm.hidden_size))
        x, h = self.lstm(x, h)
        # Flatten layer axis into batch axis so FC applies independently across layers.
        x = (self.fc(x.contiguous().view(-1, x.size(-1)))
             .view(input_size[0], input_size[1], -1))
        return x

def main():
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Read the data
    train_data = np.load(os.path.join(args.input_dir, 'train_data.npy'))
    valid_data = np.load(os.path.join(args.input_dir, 'valid_data.npy'))

    if args.n_train is not None and args.n_train > 0:
        train_data = train_data[:args.n_train]
    if args.n_valid is not None and args.n_valid > 0:
        valid_data = valid_data[:args.n_valid]

    logging.info('Loaded training data: %s' % (train_data.shape,))
    logging.info('Loaded validation data: %s' % (valid_data.shape,))

    # Inputs are the hits from [0, N-1).
    # Targets are the hits from [1, N) without the radius feature.
    torchutils.set_cuda(args.cuda)
    train_input = torchutils.np_to_torch(train_data[:,:-1])
    train_target = torchutils.np_to_torch(train_data[:,1:,:2])
    valid_input = torchutils.np_to_torch(valid_data[:,:-1])
    valid_target = torchutils.np_to_torch(valid_data[:,1:,:2])

    # Construct the model and estimator
    estimator = Estimator(
            HitPredictor(hidden_dim=args.hidden_dim),
            loss_func=nn.MSELoss(), cuda=args.cuda)

    ## Train the model
    estimator.fit(train_input, train_target,
                  valid_input=valid_input, valid_target=valid_target,
                  batch_size=args.batch_size, n_epochs=args.n_epochs)

    # Save outputs
    if args.output_dir is not None:
        logging.info('Writing outputs to %s' % args.output_dir)
        make_path = lambda s: os.path.join(args.output_dir, s)
        # Serialize the model
        torch.save(estimator.model, make_path('model'))
        # Save the losses for plotting
        np.savez(os.path.join(args.output_dir, 'losses'),
                 train_losses=estimator.train_losses,
                 valid_losses=estimator.valid_losses)

    # Drop to IPython interactive shell
    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
