#!/usr/bin/env python

# System imports
import os
import math
import logging
import argparse

# External imports
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

# Local imports
from metrics import calc_hit_accuracy
from toydata import generate_data, track_hit_coords
from drawing import (draw_layers, draw_projections, draw_3d_event,
                     draw_train_history)


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('simpleLSTM_2D')
    add_arg = parser.add_argument
    add_arg('-m', '--model', default='default',
            choices=['default', 'convae'], help='Name the model to use')
    add_arg('-n', '--num-train', type=int, default=640000,
            help='Number of events to simulate for training')
    add_arg('--num-epoch', type=int, default=10,
            help='Number of epochs in which to record training history')
    add_arg('-t', '--num-test', type=int, default=51200,
            help='Number of events to simulate for testing')
    add_arg('-o', '--output-dir',
            help='Directory to save model and plots')
    add_arg('--num-det-layer', type=int, default=10,
            help='Number of detector layers')
    add_arg('--det-layer-size', type=int, default=32,
            help='Width of the detector layers in pixels')
    add_arg('--num-seed-layer', type=int, default=3,
            help='Number of track seeding detector layers')
    add_arg('--batch-size', type=int, default=128)
    add_arg('--avg-bkg-tracks', type=int, default=3)
    add_arg('--noise-prob', type=float, default=0.01)
    return parser.parse_args()

def build_conv_model(shape):
    """Build the convolutional autoencoder model"""
    from keras import models, layers
    from keras.regularizers import l2

    inputs = layers.Input(shape=shape)

    # Need a 'channel' dimension for 3D convolution, though we have only 1 channel
    hidden = layers.Reshape((1,)+shape)(inputs)

    # 3D convolutional layers
    conv_args = dict(border_mode='same', activation='relu')
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    # Final convolution without activation
    hidden = layers.Conv3D(1, 3, 3, 3, border_mode='same')(hidden)
    # Reshape to flatten each detector layer
    hidden = layers.Reshape((shape[0], shape[1]*shape[2]))(hidden)
    # Final softmax activation
    outputs = layers.TimeDistributed(layers.Activation('softmax'))(hidden)
    # Compile the model
    model = models.Model(input=inputs, output=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model

def build_convae_model(shape, dropout=0, l2reg=0, pool=(1,2,2)):
    """Build the CNN model"""
    from keras import models, layers
    from keras.regularizers import l2

    inputs = layers.Input(shape=shape)

    # Need a 'channel' dimension for 3D convolution,
    # though we have only 1 channel initially.
    hidden = layers.Reshape((1,)+shape)(inputs)

    # 3D convolutional layers
    conv_args = dict(border_mode='same', activation='relu')
    hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.MaxPooling3D(pool_size=pool)(hidden)
    #hidden = layers.Dropout(dropout)(hidden)
    hidden = layers.Conv3D(16, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.Conv3D(16, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.MaxPooling3D(pool_size=pool)(hidden)
    #hidden = layers.Dropout(dropout)(hidden)
    #hidden = layers.Conv3D(32, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.MaxPooling3D(pool_size=pool)(hidden)
    #hidden = layers.Dropout(dropout)(hidden)
    #hidden = layers.Conv3D(64, 3, 3, 3, **conv_args)(hidden)
    #hidden = layers.MaxPooling3D(pool_size=pool)(hidden)
    #hidden = layers.Dropout(dropout)(hidden)
    #hidden = layers.Conv3D(96, 3, 2, 2, **conv_args)(hidden)
    #hidden = layers.MaxPooling3D(pool_size=pool)(hidden)
    #hidden = layers.Dropout(dropout)(hidden)
    #hidden = layers.Conv3D(128, 3, 1, 1, **conv_args)(hidden)
    # Permute dimensions to group detector layers:
    # (channels, det_layers, w, w) -> (det_layers, channels, w, w)
    PermuteLayer = layers.Permute((2, 1, 3, 4))
    hidden = PermuteLayer(hidden)
    # Reshape to flatten each detector layer: (det_layers, -1)
    perm_shape = PermuteLayer.output_shape
    flat_shape = (perm_shape[1], np.prod(perm_shape[2:]))
    hidden = layers.Reshape(flat_shape)(hidden)
    # Output softmax
    outputs = layers.TimeDistributed(
        layers.Dense(shape[1]*shape[2], activation='softmax',
                     W_regularizer=l2(l2reg))
        )(hidden)
    # Compile the model
    model = models.Model(input=inputs, output=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam', metrics=['accuracy'])
    return model

def flatten_layers(data):
    """Flattens each 2D detector layer into a 1D array"""
    return data.reshape((data.shape[0], data.shape[1], -1))

def batch_generator(num_batch, det_shape, num_seed_layers,
                    avg_bkg_tracks, noise_prob):
    """Generator of toy data batches for training"""
    shape = (num_batch,) + det_shape
    while True:
        events, sig_tracks, _ = generate_data(
            shape, num_seed_layers=num_seed_layers,
            avg_bkg_tracks=avg_bkg_tracks,
            noise_prob=noise_prob, verbose=False)
        yield (events, flatten_layers(sig_tracks))

def main():

    args = parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')

    # Configuration
    logging.info('Configuring with options: %s' % args)
    det_shape = (args.num_det_layer, args.det_layer_size, args.det_layer_size)
    logging.info('Detector shape: %s' % (det_shape,))
    
    # Random seed
    np.random.seed(2017)

    # Build the model
    logging.info('Building model')
    if args.model == 'convae':
        build_model = build_convae_model
    else:
        build_model = build_conv_model
    model = build_model(det_shape)
    model.summary()
    
    # Train the model
    logging.info('Training the model')
    events_per_epoch = args.num_train / args.num_epoch
    bgen = batch_generator(args.batch_size, det_shape, args.num_seed_layer,
                           args.avg_bkg_tracks, args.noise_prob)
    history = model.fit_generator(bgen, samples_per_epoch=events_per_epoch,
                                  nb_epoch=args.num_epoch)
    logging.info('')

    # Create a test set
    logging.info('Creating a test set')
    seed_max = 4294967295
    np.random.seed(hash('HEP.TrkX') % seed_max)
    test_events, test_tracks, test_params = generate_data(
        (args.num_test,) + det_shape, num_seed_layers=args.num_seed_layer,
        avg_bkg_tracks=args.avg_bkg_tracks, noise_prob=args.noise_prob,
        verbose=False)
    test_input = test_events
    test_target = flatten_layers(test_tracks)

    # Run model on the test set
    logging.info('Processing the test set')
    test_preds = model.predict(test_input, batch_size=args.batch_size)

    # Evaluate performance
    pixel_accuracy = calc_hit_accuracy(test_preds, test_target,
                                       num_seed_layers=args.num_seed_layer)
    # Hit classification accuracy
    test_scores = test_preds * flatten_layers(test_events)
    hit_accuracy = calc_hit_accuracy(test_scores, test_target)
    logging.info('Accuracy of predicted pixel: %g' % pixel_accuracy)
    logging.info('Accuracy of classified hit: %g' % hit_accuracy)

    if args.output_dir is not None:
        logging.info('Saving outputs to %s' % args.output_dir)

        # Plot training history
        filename = os.path.join(args.output_dir, 'training.png')
        draw_train_history(history, draw_val=False).savefig(filename)

        # Plot the first 5 events from the test set
        for i in range(5):

            event, track, params = test_events[i], test_tracks[i], test_params[i]
            pred = test_preds[i].reshape(det_shape)

            # Get the track hit coordinates
            sigx, sigy = track_hit_coords(params, np.arange(args.num_det_layer),
                                          as_type=np.float32)

            # Draw model inputs
            filename = os.path.join(args.output_dir, 'ev%i_inputs.png' % i)
            draw_layers(event, truthx=sigx, truthy=sigy).savefig(filename)
            # Draw model outputs
            filename = os.path.join(args.output_dir, 'ev%i_outputs.png' % i)
            draw_layers(pred, truthx=sigx, truthy=sigy).savefig(filename)
            # Draw input projections
            filename = os.path.join(args.output_dir, 'ev%i_inputProj.png' % i)
            draw_projections(event, truthx=sigx, truthy=sigy).savefig(filename)
            # Draw output projections
            filename = os.path.join(args.output_dir, 'ev%i_outputProj.png' % i)
            draw_projections(pred, truthx=sigx, truthy=sigy).savefig(filename)

            # Draw the 3D plot
            filename = os.path.join(args.output_dir, 'ev%i_plot3d.png' % i)
            fig, ax = draw_3d_event(event, track, params, pred,
                                    pred_threshold=0.01)
            fig.savefig(filename)

            plt.close('all')

    logging.info('All done!')

if __name__ == '__main__':
    main()
