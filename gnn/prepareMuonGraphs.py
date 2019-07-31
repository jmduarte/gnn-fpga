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

from Muon_graph import construct_graph, save_graphs

def parse_args():
    parser = argparse.ArgumentParser('prepareGraphs.py')
    add_arg = parser.add_argument
    add_arg('--input-muon-dir',
            default='/data/mliu/endcapmuons/singledata/')
    add_arg('--input-pu-dir',
            default='/data/mliu/endcapmuons/singledata_pu200/')
    add_arg('--muononly',
            default=True)
    add_arg('--max-files',
            default=1)
    add_arg('--max-events',
            default=3)
    add_arg('--output-dir',
            default="./")
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
    all_muon_files = os.listdir(args.input_muon_dir)
    all_pu_files = os.listdir(args.input_pu_dir)
    suffix = '.root'
    file_prefixes_muon = sorted(os.path.join(args.input_muon_dir, f.replace(suffix, ''))
                                for f in all_muon_files if f.endswith(suffix))
    file_prefixes_pu = sorted(os.path.join(args.input_pu_dir, f.replace(suffix, ''))
                              for f in all_pu_files if f.endswith(suffix))

    file_prefixes_muon = file_prefixes_muon[0:args.max_files]
    file_prefixes_pu = file_prefixes_pu[0:args.max_files]
    print('\njust reading up to %i files'%args.max_files)
    print('\nmuon files:', file_prefixes_muon)
    print('\npu files:', file_prefixes_pu)

    graphs = []
    for file_prefix_muon, file_prefix_pu in zip(file_prefixes_muon,file_prefixes_pu):

        print('\nreading muon file:' + file_prefix_muon + '.root')
        print('\nreading pu file:' + file_prefix_pu + '.root')

        upfile_muon = uproot.open(file_prefix_muon + '.root')
        upfile_pu = uproot.open(file_prefix_pu + '.root')
    
        print('\ntree contents:')
        # open tree, list variables
        tree_muon = upfile_muon['ntupler']['tree']
        tree_muon.show()
        tree_pu = upfile_pu['ntupler']['tree']

        # Enums
        kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4
    
        # Globals
        eta_bins = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.15, 2.5])
        eta_bins = eta_bins[::-1]
        pt_bins = np.array([-0.50, -0.333333, -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.333333, 0.50])
        nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)
    
        # get first max_events events and put it in dataframe
        print('reading only %i events'%args.max_events)
        df_muon = tree_muon.pandas.df(['vh_sim_r','vh_sim_phi','vh_sim_z','vh_sim_tp1','vh_sim_tp2',
                         'vh_type','vh_station','vh_ring','vh_sim_theta'], entrystart=0, entrystop=args.max_events)
        # filtering out events with layer number which equals -99
        df_pu = tree_pu.pandas.df(['vh_sim_r','vh_sim_phi','vh_sim_z','vh_sim_tp1','vh_sim_tp2',
                                   'vh_type','vh_station','vh_ring'], entrystart=0, entrystop=args.max_events)

        # get layer number from (vh_type, vh_station, vh_ring)
        df_muon['vh_layer'] = df_muon.apply(lambda row: get_layer(row['vh_type'], row['vh_station'], row['vh_ring']), axis=1)
        df_pu['vh_layer'] = df_pu.apply(lambda row: get_layer(row['vh_type'], row['vh_station'], row['vh_ring']), axis=1)
        df_muon = df_muon[df_muon["vh_layer"]>=0]
        df_pu = df_pu[df_pu["vh_layer"]>=0]
    
        df_muon['isMuon'] = np.ones(len(df_muon))
        df_pu['isMuon'] = np.zeros(len(df_pu))
    
        index_frame_muon = df_muon.index.to_frame()
        df_muon['event_id'] = index_frame_muon['entry']
    
        index_frame_pu = df_pu.index.to_frame()
        df_pu['event_id'] = index_frame_pu['entry']
    
        print('\nmuon events:', df_muon)
        print('\npu events:', df_pu)
    
        # get only true muon hits (generator-level matching condition)!
        df_muon = df_muon[(df_muon['vh_sim_tp1']==0) & (df_muon['vh_sim_tp2']==0)]   
        frames_muon = []
        frames_pu = []
        frames_all = []
        hits = []
        hit_distr = [0]*12

        for entry, new_df_muon in df_muon.groupby(level=0):
            new_df_muon = new_df_muon.drop_duplicates(['vh_type','vh_station','vh_ring'])
            frames_muon.append(new_df_muon)
            # Count hits/muon
            hit = new_df_muon.shape[0]
            hits.append(hit)
        for entry_pu, new_df_pu in df_pu.groupby(level=0):
            frames_pu.append(new_df_pu)
            new_df_all = pd.concat([new_df_pu, frames_muon[entry_pu]]) 
            frames_all.append(new_df_all)
            # Count number hits/layer
            for row in new_df_pu.itertuples():
                layr = getattr(row, "vh_layer") 
                hit_distr[layr] += 1   
    
        df_muon = pd.concat(frames_muon)
    
        df_all = pd.concat(frames_all)

        hit_distr = [i/41 for i in hit_distr] # If 41 = entrystop
        print('hit distr:', hit_distr)
        
        # Define adjacent layers
        n_det_layers = 12
        l = np.arange(n_det_layers)
        layer_pairs = np.stack([l[:-1], l[1:]], axis=1)
    
        #feature_names = ['vh_sim_r', 'vh_sim_phi', 'vh_sim_z']
        #n_phi_sectors = 1
        #feature_scale = np.array([1000., np.pi / n_phi_sectors, 1000.])
        feature_names = ['vh_sim_z', 'vh_sim_theta','vh_layer' ,'vh_sim_phi','vh_sim_r']
        n_phi_sectors = 6
        feature_scale = np.array([1.,1 ,1 ,np.pi / n_phi_sectors, 1.])
      
        df = df_all
        if args.muononly: df = df_muon
        for entry_all, new_df_all in df.groupby(level=0):
            graph = [construct_graph(new_df_all, layer_pairs=layer_pairs,
                                     feature_names=feature_names,
                                     feature_scale=feature_scale)]
            graphs.append(graph)
        
        # Write outputs
        graphs = [g for gs in graphs for g in gs]
        if args.output_dir:
           os.system('mkdir -p %s'%args.output_dir)
           logging.info('Writing outputs to ' + args.output_dir)
        # Write out the graphs
        filenames = [os.path.join(args.output_dir, 'graph%06i' % i)
                     for i in range(len(graphs))]
        save_graphs(graphs, filenames)
    return graphs

if __name__ == '__main__':
    main()
