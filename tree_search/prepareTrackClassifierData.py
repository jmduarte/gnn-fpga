#!/usr/bin/env python

# System imports
import os
import logging
import argparse
import multiprocessing as mp

# External imports
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from data import process_files
from track_filter import (coord_scale, select_hits,
                          select_signal_hits, remove_duplicate_hits_2)

def parse_args():
    parser = argparse.ArgumentParser('prepareData.py')
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu10_pt1000_2017_07_29')
    add_arg('--output-dir')
    add_arg('--n-files', type=int, default=1)
    add_arg('--n-workers', type=int, default=1)
    add_arg('--valid-frac', type=float, default=0.1)
    add_arg('--test-frac', type=float, default=0.1)
    add_arg('--show-config', action='store_true',
            help='Dump the command line config')
    add_arg('--interactive', action='store_true',
            help='Drop into IPython shell at end of script')
    return parser.parse_args()

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def calc_dphi(phi1, phi2):
    """TODO: update this for signed dph"""
    dphi = np.abs(phi1 - phi2)
    idx = dphi > np.pi
    dphi[idx] = 2*np.pi - dphi[idx]
    return dphi

def calc_eta_phi_distance(r1, phi1, z1, r2, phi2, z2):
    # First, calculate the eta coordinates
    eta1 = calc_eta(r1, z1)
    eta2 = calc_eta(r2, z2)
    deta = np.abs(eta1 - eta2)
    dphi = calc_dphi(phi1, phi2)
    return np.sqrt(dphi*dphi + deta*deta)

def standardize_features(features):
    # Center every track in phi on the first hit.
    features[:,:,0] -= features[:,:1,0]
    features[features[:,:,0] > np.pi, 0] -= 2*np.pi
    features[features[:,:,0] < -np.pi, 0] += 2*np.pi
    # Scale coordinates
    features[:,:] /= coord_scale
    return features

def select_samples(hits):
    # Group hits by event
    event_groups = hits.groupby('evtid')
    # Select true particle samples
    signal_hits = select_signal_hits(hits)
    sample_keys = signal_hits[['evtid', 'barcode']].drop_duplicates().values
    n_samples = sample_keys.shape[0]
    logging.info('Selected %i track samples' % n_samples)

    # Just hardcode for now
    n_det_layers = 10
    seed_size = 3
    epsilon = 0.05
    feature_names = ['phi', 'z', 'r']
    n_features = len(feature_names)

    # Features and labels arrays
    features = np.zeros((n_samples, n_det_layers, n_features), dtype=np.float32)
    labels = np.zeros((n_samples, n_det_layers), dtype=np.float32)

    # Loop over samples
    for i, k in enumerate(sample_keys):
    
        # Gather the hits for this sample
        sample_hits = event_groups.get_group(k[0]).sort_values('layer')
        sample_labels = sample_hits.barcode == k[1]
        # Prepare the seed
        seed_hits = sample_hits[(sample_hits.layer < seed_size) &
                                (sample_hits.barcode == k[1])]
        features[i, :seed_size] = seed_hits[feature_names].values
        #indices[i, :seed_size] = seed_hits.index
        labels[i, :seed_size] = 1
        real_track = True
    
        # Loop over layers
        for layer in range(seed_size, n_det_layers):
            # Select all candidate hits on the layer
            layer_hits = sample_hits[sample_hits.layer == layer]
            layer_labels = sample_labels.loc[layer_hits.index]
            # Extract the true hit
            true_hit = layer_hits[layer_labels].iloc[0]
            # Calculate distances of hits from the true hit
            dr = calc_eta_phi_distance(layer_hits.r, layer_hits.phi, layer_hits.z,
                                       true_hit.r, true_hit.phi, true_hit.z)
            # Calculate proximity weights regularized by epsilon
            w = 1 / (dr + epsilon)
            # Sample one hit from the candidates using proximity weights.
            sampled_hit = layer_hits.sample(weights=w)
            sampled_idx = sampled_hit.index[0]
            #indices[i, layer] = sampled_idx
            features[i, layer] = sampled_hit[feature_names]
            # If a previous hit was wrong the track is always considered 'fake'
            real_track = real_track and layer_labels.loc[sampled_idx]
            labels[i, layer] = real_track

    # Standardize the features
    features = standardize_features(features)
    # Return the filled arrays
    return features, labels

def main():
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Read the data
    all_files = os.listdir(args.input_dir)
    hits_files = sorted(f for f in all_files if f.startswith('clusters'))
    hits_files = [os.path.join(args.input_dir, f)
                  for f in hits_files[:args.n_files]]
    hits = process_files(hits_files, num_workers=args.n_workers, concat=False)
    logging.info('Loaded hits data with shapes: %s' % (map(np.shape, hits),))

    # Select good track hits using process pool
    pool = mp.Pool(processes=args.n_workers)
    hits = pool.map(select_hits, hits)
    hits = pool.map(remove_duplicate_hits_2, hits)

    # Select track samples
    track_samples = pool.map(select_samples, hits)
    features = np.concatenate([s[0] for s in track_samples])
    labels = np.concatenate([s[1] for s in track_samples])
    logging.info('Features: %s' % (features.shape,))
    logging.info('Labels: %s' % (labels.shape,))

    # Close the process pool
    pool.close()
    pool.join()

    # Split into training, validation, and test sets
    val_test_frac = args.test_frac + args.valid_frac
    train_features, valtest_features, train_labels, valtest_labels = (
        train_test_split(features, labels, test_size=val_test_frac))
    test_sub_frac = args.test_frac / val_test_frac
    valid_features, test_features, valid_labels, test_labels = (
        train_test_split(valtest_features, valtest_labels, test_size=test_sub_frac))
    logging.info('Train features: %s' % (train_features.shape,))
    logging.info('Train labels: %s' % (train_labels.shape,))
    logging.info('Test features: %s' % (test_features.shape,))
    logging.info('Test labels: %s' % (test_labels.shape,))

    # Save the results
    if args.output_dir is not None:
        logging.info('Writing data to %s' % args.output_dir)
        np.save(os.path.join(args.output_dir, 'train_features'), train_features)
        np.save(os.path.join(args.output_dir, 'train_labels'), train_labels)
        np.save(os.path.join(args.output_dir, 'valid_features'), valid_features)
        np.save(os.path.join(args.output_dir, 'valid_labels'), valid_labels)
        np.save(os.path.join(args.output_dir, 'test_features'), test_features)
        np.save(os.path.join(args.output_dir, 'test_labels'), test_labels)

    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
