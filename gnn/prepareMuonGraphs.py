#!/usr/bin/env python

"""
This script is used to construct the muon graph samples for
input into the models.
"""

from __future__ import print_function
from __future__ import division 

import os
import logging
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import uproot

def parse_args():
    parser = argparse.ArgumentParser('prepareGraphs.py')
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/data/jduarte1/muon/')
    return parser.parse_args()



# Decide EMTF hit layer number
emtf_lut = np.zeros((5,5,5), dtype=np.int32) - 99
emtf_lut[1,1,4] = 0  # ME1/1a
emtf_lut[1,1,1] = 0  # ME1/1b
emtf_lut[1,1,2] = 1  # ME1/2
emtf_lut[1,1,3] = 1  # ME1/3
emtf_lut[1,2,1] = 2  # ME2/1
emtf_lut[1,2,2] = 2  # ME2/2
emtf_lut[1,3,1] = 3  # ME3/1
emtf_lut[1,3,2] = 3  # ME3/2
emtf_lut[1,4,1] = 4  # ME4/1
emtf_lut[1,4,2] = 4  # ME4/2
emtf_lut[2,1,2] = 5  # RE1/2
emtf_lut[2,2,2] = 6  # RE2/2
emtf_lut[2,3,1] = 7  # RE3/1
emtf_lut[2,3,2] = 7  # RE3/2
emtf_lut[2,3,3] = 7  # RE3/3
emtf_lut[2,4,1] = 8  # RE4/1
emtf_lut[2,4,2] = 8  # RE4/2
emtf_lut[2,4,3] = 8  # RE4/3
emtf_lut[3,1,1] = 9  # GE1/1
emtf_lut[3,2,1] = 10 # GE2/1
emtf_lut[4,1,1] = 11 # ME0

def get_layer(type, station, ring):
    return emtf_lut[int(type),int(station),int(ring)]

def main():
    """Main program function"""
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')

    # Find the input files
    all_files = os.listdir(args.input_dir)
    suffix = '.root'
    file_prefixes = sorted(os.path.join(args.input_dir, f.replace(suffix, ''))
                           for f in all_files if f.endswith(suffix))

    print('\nfiles:', file_prefixes)
    # just read first file (muon) for now
    upfile = uproot.open(file_prefixes[0] + '.root')

    
    print('\ntree:', file_prefixes)
    # open tree, list variables
    tree = upfile['ntupler']['tree']
    tree.show()


    # Enums
    kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4
    
    # Globals
    eta_bins = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.15, 2.5])
    eta_bins = eta_bins[::-1]
    pt_bins = np.array([-0.50, -0.333333, -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.333333, 0.50])
    nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)
    
    # get first 10 events and put it in dataframe
    df = tree.pandas.df(['vh_sim_r','vh_sim_phi','vh_sim_z','vh_sim_tp1','vh_sim_tp2',
                         'vh_type','vh_station','vh_ring'], entrystart=0, entrystop=10)

    # get layer number from (vh_type, vh_station, vh_ring)
    df['vh_layer'] = df.apply(lambda row: get_layer(row['vh_type'], row['vh_station'], row['vh_ring']), axis=1)

    print('\nevents:', df)


if __name__ == '__main__':
    main()
