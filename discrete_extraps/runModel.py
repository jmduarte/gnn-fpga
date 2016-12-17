#!/usr/bin/env python

# System imports
import math
import logging
import argparse

# External imports
import numpy as np
from keras import models
from keras import layers

# Local imports
from metrics import calc_hit_accuracy
from models import build_lstm_model, build_deep_lstm_model

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('simpleLSTM_2D')
    add_arg = parser.add_argument
    add_arg('-m', '--model', default='default', choices=['default', 'deep'],
            help='Name the model to use')
    add_arg('-n', '--num-event', type=int, default=100000,
            help='Number of events to simulate')
    add_arg('-o', '--output-dir',
            help='Directory to save model and training history')
    add_arg('--write-history', action='store_true',
            help='Write the history object to output directory')
    add_arg('--num-det-layer', type=int, default=10,
            help='Number of detector layers')
    add_arg('--det-layer-size', type=int, default=10,
            help='Width of the detector layers in pixels')
    add_arg('--num-seed-layer', type=int, default=3,
            help='Number of track seeding detector layers')
    add_arg('--num-hidden', type=int, default=100)
    add_arg('--batch-size', type=int, default=500)
    add_arg('--num-epoch', type=int, default=20)
    add_arg('--valid-frac', type=int, default=0.2)
    add_arg('--avg-bkg-tracks', type=int, default=2)
    add_arg('--noise-prob', type=float, default=0.01)
    return parser.parse_args()

def gen_noise_2d(shape, prob=0.1, seed_layers=0):
    noise = (np.random.random_sample(shape) < prob).astype(np.int8)
    noise[:,:seed_layers,:,:] = 0
    return noise

def gen_straight_tracks_2d(n, num_layers, layer_size):
    # Initialize the data
    data = np.zeros((n, num_layers, layer_size, layer_size),
                    dtype=np.float32)
    # Sample the entry and exit points for tracks
    entry_points = np.random.uniform(0, layer_size, size=(n, 2))
    exit_points = np.random.uniform(0, layer_size, size=(n, 2))
    # Calculate slope parameters
    slopes = (exit_points - entry_points) / float(num_layers - 1)
    # Calculate hit positions and fill hit data
    xhits = np.zeros(num_layers, dtype=np.int)
    yhits = np.zeros(num_layers, dtype=np.int)
    idx = np.arange(num_layers)
    for ievt in range(n):
        xhits[:] = slopes[ievt,0]*idx + entry_points[ievt,0]
        yhits[:] = slopes[ievt,1]*idx + entry_points[ievt,1]
        data[ievt,idx,xhits,yhits] = 1
    return data

def gen_bkg_tracks_2d(num_event, num_layers, layer_size,
                      avg_bkg_tracks=3, seed_layers=0):
    num_bkg_tracks = np.random.poisson(avg_bkg_tracks, num_event)
    bkg_tracks = np.zeros((num_event, num_layers, layer_size, layer_size),
                          dtype=np.float32)
    for ievt in range(num_event):
        ntrk = num_bkg_tracks[ievt]
        bkg_tracks[ievt] = sum(gen_straight_tracks_2d(ntrk, num_layers, layer_size))
    bkg_tracks[:,:seed_layers,:,:] = 0
    return bkg_tracks

def flatten_layers(data):
    """Flattens each 2D detector layer into a 1D array"""
    return data.reshape((data.shape[0], data.shape[1], -1))

def flat_to_2d(data, det_width):
    """Expands the flattened layers to original (width x width)"""
    return data.reshape((data.shape[0], data.shape[1], det_width, det_width))

def main():

    args = parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')

    # Configuration
    logging.info('Configuring with options: %s' % args)
    shape = (args.num_event, args.num_det_layer,
             args.det_layer_size, args.det_layer_size)
    det_shape = shape[1:]
    logging.info('Data shape: %s' % (shape,))
    
    # Data generation
    # Signal tracks
    np.random.seed(2017)
    sig_tracks = gen_straight_tracks_2d(args.num_event, args.num_det_layer,
                                        args.det_layer_size)
    # Background tracks
    bkg_tracks = gen_bkg_tracks_2d(args.num_event, args.num_det_layer,
                                   args.det_layer_size,
                                   avg_bkg_tracks=args.avg_bkg_tracks,
                                   seed_layers=args.num_seed_layer)
    # Noise
    noise = gen_noise_2d(shape, prob=args.noise_prob, seed_layers=args.num_seed_layer)
    
    # Full events
    events = sig_tracks + bkg_tracks + noise
    events[events > 1] = 1

    # Print data sizes
    logging.info('Sizes of arrays')
    size = lambda x: x.dtype.itemsize * x.size / 1e6
    logging.info('  events:     %g MB' % (size(events)))
    logging.info('  sig_tracks: %g MB' % (size(sig_tracks)))
    logging.info('  bkg_tracks: %g MB' % (size(bkg_tracks)))
    logging.info('  noise:      %g MB' % (size(noise)))
    logging.info('  checksum:   %i' % events.sum())
    
    # Clean up
    del bkg_tracks
    del noise
    
    # Setup the inputs and outputs
    train_input = flatten_layers(events[:,:-1,:,:])
    train_target = flatten_layers(sig_tracks[:,1:,:,:])
    
    # Build the model
    logging.info('Building model')
    if args.model == 'default':
        model_func = build_lstm_model
    elif args.model == 'deep':
        model_func = build_deep_lstm_model
    else:
        raise Exception('Unknown requested model: %s' % args.model)
    model = model_func(args.num_det_layer-1, args.det_layer_size**2,
                       hidden_dim=args.num_hidden)
    model.summary()
    
    # Train the model
    logging.info('Training the model')
    history = model.fit(train_input, train_target,
                        batch_size=args.batch_size, nb_epoch=args.num_epoch,
                        validation_split=args.valid_frac)
    logging.info('')

    # Get all of the training data predictions
    train_preds = model.predict(train_input, batch_size=args.batch_size)

    #if args.output_dir is not None:
    #    filename = os.path.join([args.output_dir, 'model_' + args.model + '.h5'])
    #    logging.info('Saving model to %s' % filename)
    #    model.save(filename)

    #    if args.write_history:
    #        history_filename = os.path.join([args.output_dir, 'history.npz'])
    #        logging.info('Saving training history to %s' % history_filename)
    #        np.savez(history_filename, **history.history)
    
    # Report metrics
    # Is this correct or am I neglecting one layer?
    acc = calc_hit_accuracy(train_preds, train_target,
                            num_seed_layers=args.num_seed_layer)

    # Hit classification accuracy
    hit_scores = train_preds * flatten_layers(events[:,1:,:,:])
    class_acc = calc_hit_accuracy(hit_scores, train_target)

    logging.info('Model accuracy: %f' % acc)
    logging.info('Classification accuracy: %f' % class_acc)
    logging.info('All done!')

if __name__ == '__main__':
    main()
