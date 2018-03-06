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
from track_filter import HitPredictor, HitGausPredictor, gaus_llh_loss

def parse_args():
    parser = argparse.ArgumentParser('trainTrackFilter.py')
    add_arg = parser.add_argument
    add_arg('--input-dir', default='/global/cscratch1/sd/sfarrell/heptrkx/RNNFilter')
    add_arg('--continue-dir',
            help='Directory containing model to continue training')
    add_arg('--output-dir')
    add_arg('--model', choices=['regression', 'gaus'], default='regression')
    add_arg('--n-train', type=int, help='Maximum number of training samples')
    add_arg('--n-valid', type=int, help='Maximum number of validation samples')
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--batch-size', type=int, default=32)
    add_arg('--hidden-dim', type=int, default=20)
    add_arg('--cuda', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

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

    model = None
    train_losses, valid_losses = [], []

    # If we're continuing a pre-trained model, load it up now
    if args.continue_dir is not None and args.continue_dir != '':
        logging.info('Loading model to continue training from %s' % args.continue_dir)
        model_file = os.path.join(args.continue_dir, 'model')
        model = torch.load(model_file)
        losses_file = os.path.join(args.continue_dir, 'losses.npz')
        losses_data = np.load(losses_file)
        train_losses = list(losses_data['train_losses'])
        valid_losses = list(losses_data['valid_losses'])

    # Configure model type and loss function
    if args.model == 'regression':
        model_type = HitPredictor
        loss_func = nn.MSELoss()
    else:
        model_type = HitGausPredictor
        loss_func = gaus_llh_loss

    # Construct the model if not done already
    if model is None:
        model = model_type(hidden_dim=args.hidden_dim)
    # Construct the estimator
    estimator = Estimator(model, loss_func=loss_func, cuda=args.cuda,
                          train_losses=train_losses, valid_losses=valid_losses)

    # Train the model
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
