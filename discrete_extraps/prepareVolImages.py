#!/usr/bin/env python

"""
This script contains code for converting ACTS hit data into
per-detector-volume 3D binned representations.
"""

# Python 2-3 compatibility
from __future__ import print_function
from __future__ import division

# System imports
import sys
import os
import logging
import argparse
import multiprocessing as mp
from functools import partial

# External imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from data import (process_files, select_barrel_hits,
                  check_data_consistency, bin_barrel_hits)

# Some global, hardcoded config, for current convenience
vols = [0, 1, 2]
n_vols = len(vols)
# Number of bins defined per volume as (numLayer, numPhi, numZ)
bins = [
    [4, 128, 128],
    [4, 128, 128],
    [2, 128, 21]
]
# Ranges similarly defined per volume as (rangeLayer, rangePhi, rangeZ)
ranges = [
    [[0, 4], [-np.pi, np.pi], [-500, 500]],
    [[4, 8], [-np.pi, np.pi], [-1080, 1080]],
    [[8, 10], [-np.pi, np.pi], [-1031, 1031]]
]

def parse_args():
    """Parse the command line options"""
    parser = argparse.ArgumentParser(sys.argv[0])
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu10_pt1000_2017_07_29')
    add_arg('--output-dir')
    add_arg('--n-files', type=int, default=1)
    add_arg('--n-workers', type=int, default=1)
    add_arg('--test-frac', type=float, default=0.1)
    add_arg('--show-config', action='store_true',
            help='Dump the command line config')
    add_arg('--interactive', action='store_true',
            help='Drop into IPython shell at end of script')
    return parser.parse_args()

def compute_labels(hits, particles, evtids,
                   min_layers=7, min_pt=2, min_tracks=3):
    """Compute sample labels according to the trigger criteria"""
    # Compute number of layers hit for each truth particle
    join_keys = ['evtid', 'barcode']
    nlayer = (hits.groupby(join_keys).apply(lambda x: len(x.layer.unique()))
              .reset_index(name='nlayer'))
    pars = particles.merge(nlayer, on=join_keys)
    # Compute the trigger decision labels
    trig_func = lambda x: ((x.nlayer >= min_layers) & (x.pt > min_pt)).sum() >= min_tracks
    trigger_results = pars.groupby('evtid').apply(trig_func)
    return trigger_results.loc[evtids].values.astype(np.float32)

def main():
    """Main execution function"""
    # Parse command line
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Find the data files
    all_files = os.listdir(args.input_dir)
    hits_files = sorted(f for f in all_files if f.startswith('clusters'))
    hits_files = [os.path.join(args.input_dir, f) for f in hits_files[:args.n_files]]
    particles_files = sorted(f for f in all_files if f.startswith('particles'))
    particles_files = [os.path.join(args.input_dir, f) for f in particles_files[:args.n_files]]

    # Start worker process pool
    with mp.Pool(processes=args.n_workers) as pool:
        # Load the data
        hits, particles = process_files(hits_files, particles_files, pool)
        # Remove files that have data matching problems
        hits, particles = check_data_consistency(hits, particles)
        logging.info('Loaded hits shapes: %s' % (list(map(np.shape, hits),)))
        # Apply data selection
        hits = pool.map(select_barrel_hits, hits)
        logging.info('Selected hits shapes: %s' % (list(map(np.shape, hits),)))
        # Extract the list of event IDs to process
        evtids = [h.evtid.unique() for h in hits]
        # Compute trigger classification labels
        logging.info('Computing trigger classification labels')
        labels = pool.starmap(compute_labels, zip(hits, particles, evtids))
        # Construct per-detector-volume images
        logging.info('Constructing detector images')
        bin_func = partial(bin_barrel_hits, vols=vols, bins=bins, ranges=ranges)
        hists = pool.starmap(bin_func, zip(hits, evtids))

    # Combine all the data
    evtids = np.concatenate(evtids)
    hits = pd.concat(hits)
    particles = pd.concat(particles)
    labels = np.concatenate(labels)
    hists = [np.concatenate([h[iv] for h in hists]) for iv in range(n_vols)]

    logging.info('Merged data shapes:')
    logging.info('Hits: %s' % (hits.shape,))
    logging.info('Particles: %s' % (particles.shape,))
    logging.info('Labels: %s' % (labels.shape,))
    for i, h in enumerate(hists):
        logging.info('Images vol %i: %s' % (i, h.shape))

    # Split into train and test sets
    train_labels, test_labels, *split_hists = train_test_split(
        labels, *hists, test_size=args.test_frac)
    train_hists, test_hists = split_hists[::2], split_hists[1::2]

    # Save the results
    if args.output_dir is not None:
        logging.info('Writing data to %s' % args.output_dir)
        # Save the labels
        np.save(os.path.join(args.output_dir, 'train_labels'), train_labels)
        np.save(os.path.join(args.output_dir, 'test_labels'), test_labels)
        # Save the images
        for iv in range(n_vols):
            np.save(os.path.join(args.output_dir, 'train_v%i' % vols[iv]), train_hists[iv])
            np.save(os.path.join(args.output_dir, 'test_v%i' % vols[iv]), test_hists[iv])

    # Drop to interactive shell
    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
