#!/usr/bin/env python

# System imports
import os
import logging
import argparse

# External imports
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from data import process_files
from track_filter import coord_scale, select_hits, select_signal_hits

def parse_args():
    parser = argparse.ArgumentParser('prepareData.py')
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu10_pt1000_2017_07_29')
    add_arg('--output-dir')
    add_arg('--n-files', type=int, default=1)
    add_arg('--n-file-workers', type=int, default=1)
    add_arg('--test-frac', type=float, default=0.1)
    add_arg('--show-config', action='store_true',
            help='Dump the command line config')
    add_arg('--interactive', action='store_true',
            help='Drop into IPython shell at end of script')
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
    all_files = os.listdir(args.input_dir)
    hits_files = sorted(f for f in all_files if f.startswith('clusters'))
    hits_files = [os.path.join(args.input_dir, f)
                  for f in hits_files[:args.n_files]]
    hits = process_files(hits_files, num_workers=args.n_file_workers)
    logging.info('Loaded hits data with shape: %s' % (hits.shape,))

    # Select good track hits
    hits = select_signal_hits(select_hits(hits))
    logging.info('Selected hits: %s' % (hits.shape,))

    # Gather into tensor of shape (events, layers, features)
    input_data = (np.stack(hits.groupby(['evtid', 'barcode'])
                           .apply(lambda x: x[['phi', 'z', 'r']].values))
                  .astype(np.float32))
    # Scale coordinates
    input_data[:,:] /= coord_scale

    # Split into training and test sets
    train_data, test_data = train_test_split(input_data, test_size=args.test_frac)

    # Save the results
    if args.output_dir is not None:
        logging.info('Writing data to %s' % args.output_dir)
        np.save(os.path.join(args.output_dir, 'train_data'), train_data)
        np.save(os.path.join(args.output_dir, 'test_data'), test_data)

    if args.interactive:
        import IPython
        IPython.embed()
    logging.info('All done!')

if __name__ == '__main__':
    main()
