#!/usr/bin/env python

"""
This script is used to construct the muon graph samples for
input into the models.
"""

from __future__ import print_function
from __future__ import division 

import os,glob
import logging
import argparse
import multiprocessing as mp
from functools import partial
from scipy.sparse import csr_matrix, find

import numpy as np
import pandas as pd
import uproot

from Muon_graph import construct_graph, save_graphs,SparseGraphProp, load_graph,  \
				  load_graph, graph_from_sparse_prop, \
				  draw_sample_withproperties

def parse_args():
    parser = argparse.ArgumentParser('prepareGraphs.py')
    add_arg = parser.add_argument
    add_arg('--input-muon-dir',
            default='/data/mliu/endcapmuons/')
            #default='/data/mliu/endcapmuons/singledata/')
    add_arg('--input-pu-dir',
            default='/data/mliu/endcapmuons/')
            #default='/data/mliu/endcapmuons/singledata_pu200/')
    add_arg('--muononly',
            default=False)
    add_arg('--max_files',
            default=1)
    add_arg('--max_events',
            default=1)
    add_arg('--start',
            default=0)
    add_arg('--end',
            default=100)
    add_arg('--output-dir',
            default="./")
    return parser.parse_args()

# Decide EMTF hit layer number
emtf_lut = np.zeros((5,5,5), dtype=np.int32) - 99
#### emtf_lut[1,1,4] = 0  # ME1/1a
#### emtf_lut[1,1,1] = 0  # ME1/1b
#### emtf_lut[1,1,2] = 1  # ME1/2
#### emtf_lut[1,1,3] = 1  # ME1/3
#### emtf_lut[1,2,1] = 2  # ME2/1
#### emtf_lut[1,2,2] = 2  # ME2/2
#### emtf_lut[1,3,1] = 3  # ME3/1
#### emtf_lut[1,3,2] = 3  # ME3/2
#### emtf_lut[1,4,1] = 4  # ME4/1
#### emtf_lut[1,4,2] = 4  # ME4/2
#### emtf_lut[2,1,2] = 5  # RE1/2
#### emtf_lut[2,2,2] = 6  # RE2/2
#### emtf_lut[2,3,1] = 7  # RE3/1
#### emtf_lut[2,3,2] = 7  # RE3/2
#### emtf_lut[2,3,3] = 7  # RE3/3
#### emtf_lut[2,4,1] = 8  # RE4/1
#### emtf_lut[2,4,2] = 8  # RE4/2
#### emtf_lut[2,4,3] = 8  # RE4/3
#### emtf_lut[3,1,1] = 9  # GE1/1
#### emtf_lut[3,2,1] = 10 # GE2/1
#### emtf_lut[4,1,1] = 11 # ME0
emtf_lut[1,1,4] = 3  # ME1/1a
emtf_lut[1,1,1] = 3  # ME1/1b
emtf_lut[1,1,2] = 4  # ME1/2
emtf_lut[1,1,3] = 4  # ME1/3
emtf_lut[1,2,1] = 8  # ME2/1
emtf_lut[1,2,2] = 8  # ME2/2
emtf_lut[1,3,1] = 9  # ME3/1
emtf_lut[1,3,2] = 9  # ME3/2
emtf_lut[1,4,1] = 11  # ME4/1
emtf_lut[1,4,2] = 11  # ME4/2
emtf_lut[2,1,2] = 5  # RE1/2
emtf_lut[2,2,2] = 6  # RE2/2
emtf_lut[2,3,1] = 10  # RE3/1
emtf_lut[2,3,2] = 10  # RE3/2
emtf_lut[2,3,3] = 10  # RE3/3
emtf_lut[2,4,1] = 12  # RE4/1
emtf_lut[2,4,2] = 12  # RE4/2
emtf_lut[2,4,3] = 12  # RE4/3
emtf_lut[3,1,1] = 2  # GE1/1
emtf_lut[3,2,1] = 7 # GE2/1
emtf_lut[4,1,1] = 1 # ME0

def column(matrix, i):
    return [int(row[i]) for row in matrix]
def get_layer(type, station, ring):
    return emtf_lut[int(type),int(station),int(ring)]
def get_layers(types,stations,rings):
	array = []
	for i in range(0, len(types)):
	    array.append([get_layer(types[i],stations[i],rings[i])]  )
	return array
def plotgraph(file,outputname,output):
	g1sparse  = load_graph('%s'%file,graph_type=SparseGraphProp)
	g1        = graph_from_sparse_prop(g1sparse)
	#Here you can cut on the pt and eta of the generated muons and plot
	#if g1.pt>0 and abs(g1.eta)<2.4 and abs(g1.eta)>1.2:
	draw_sample_withproperties(g1.X,g1.Ri,g1.Ro,g1.y,g1.pt,g1.eta,skip_false_edges=False,outputdir=outputname,output=output)

def plotgraphs(inputdir):
	outputdir = os.path.join(inputdir, "plots")
	if not os.path.exists(outputdir): os.makedirs(outputdir)	
	files = glob.glob("%s/*.npz"%inputdir)
	for file in files: 
		plotgraph(file,outputdir,str(file.strip().split('/')[-1]).replace('.npz',''))

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
    suffix = 'SingleMuon_Endcap.root'
    file_prefixes_muon = sorted(os.path.join(args.input_muon_dir,f.replace(suffix, 'SingleMuon_Endcap'))
                                for f in all_muon_files if f.endswith(suffix))
    file_prefixes_pu = sorted(os.path.join(args.input_pu_dir, f.replace(suffix, 'SingleNeutrino_PU200'))
                              for f in all_pu_files if f.endswith(suffix))

    file_prefixes_muon = file_prefixes_muon[0:int(args.max_files)]
    file_prefixes_pu = file_prefixes_pu[0:int(args.max_files)]

    print('\njust reading up to %i files'%int(args.max_files))
    print('\nmuon files:', file_prefixes_muon)
    print('\npu files:', file_prefixes_pu)

    graphs = []

    for file_prefix_muon, file_prefix_pu in zip(file_prefixes_muon,file_prefixes_pu):

        print('\nreading muon file:' + file_prefix_muon + '.root')
        print('\nreading pu file:' + file_prefix_pu + '.root')

        upfile_muon = uproot.open(file_prefix_muon + '.root')
        upfile_pu = uproot.open(file_prefix_pu + '.root')
    
#        print('\ntree contents:')
        # open tree, list variables
        tree_muon = upfile_muon['ntupler']['tree']
#        tree_muon.show()
        tree_pu = upfile_pu['ntupler']['tree']

        # Enums
        kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4
    
        # Globals
        eta_bins = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.15, 2.5])
        eta_bins = eta_bins[::-1]
        pt_bins = np.array([-0.50, -0.333333, -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.333333, 0.50])
        nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)
    
        # get first max_events events and put it in dataframe
        print('reading only %i events'%int(args.max_events))
        hit_features= ['vh_sim_z','vh_sim_theta','vh_sim_phi','vh_sim_r','vh_bend','vh_sim_tp1',
                         'vh_sim_tp2','vh_station','vh_ring','vh_type']
        df_muon = tree_muon.pandas.df(hit_features, entrystart=int(args.start), entrystop=int(args.end))
        df_pu = tree_pu.pandas.df(hit_features, entrystart=int(args.start), entrystop=int(args.end))
        df_muon_vp = tree_muon.pandas.df(['vp_pt','vp_eta'], entrystart=int(args.start), entrystop=int(args.end))
        # get layer number from (vh_type, vh_station, vh_ring)
        df_muon['vh_layer'] = df_muon.apply(lambda row: get_layer(row['vh_type'], row['vh_station'], row['vh_ring']), axis=1)
        df_pu['vh_layer'] = df_pu.apply(lambda row: get_layer(row['vh_type'], row['vh_station'], row['vh_ring']), axis=1)
        # filtering out events with layer number which equals -99
        df_muon = df_muon[(df_pu["vh_layer"]>-99) & (df_muon["vh_layer"]>-99)]
        df_pu = df_pu[(df_pu["vh_layer"]>-99) & (df_muon["vh_layer"]> -99)]
        
        df_muon['isMuon'] = np.ones(len(df_muon))
        df_pu['isMuon'] = np.zeros(len(df_pu))
        index_frame_muon = df_muon.index.to_frame()
        df_muon['event_id'] = index_frame_muon['entry']
        index_frame_pu = df_pu.index.to_frame()
        df_pu['event_id'] = index_frame_pu['entry']
        # separate plus and minus
        #print(df_muon['vh_layer'])
        df_muon['vh_layer'] = np.multiply(df_muon['vh_layer'],np.sign(df_muon['vh_sim_z']))
        df_pu['vh_layer'] = np.multiply(df_pu['vh_layer'],np.sign(df_pu['vh_sim_z']))
        # get only true muon hits (generator-level matching condition)!
        df_muon = df_muon[(df_muon['vh_sim_tp1']==0) & (df_muon['vh_sim_tp2']==0)]   
        df_pu_group = df_pu.groupby(level=0)
        pumap_index = list(df_pu_group.groups)
        frames_muon = []
        frames_pu = []
        frames_all = []
        hits = []
        hit_distr = [0]*12
        # fill in muon hits 
        for entry, new_df_muon in df_muon.groupby(level=0):
            new_df_muon = new_df_muon.drop_duplicates(['vh_type','vh_station','vh_ring'])
            frames_muon.append(new_df_muon)
            # Count hits/muon
            hit = new_df_muon.shape[0]
            hits.append(hit)
        # mixing pu hits in
        for entry_pu, new_df_pu in df_pu.groupby(level=0):
            new_df_pu = new_df_pu.drop_duplicates(['vh_type','vh_station','vh_ring'])
            frames_pu.append(new_df_pu)
            entry_mu = pumap_index.index(entry_pu)
            if(entry_mu>=len(hits)): continue
            #return
            new_df_all = pd.concat([new_df_pu, frames_muon[entry_mu]]) 
            frames_all.append(new_df_all)
            # Count number hits/layer
            # for row in new_df_pu.itertuples():
                #layr = getattr(row, "vh_layer") 
                #hit_distr[layr] += 1   
        df_muon = pd.concat(frames_muon)
        df_all = pd.concat(frames_all)
        hit_distr = [i/41 for i in hit_distr] # If 41 = entrystop
        # layers in prepare muon graph
        # add layer to the features saved
        hit_features.append('vh_layer')
        n_phi_sectors = 6
        feature_scale = np.array([1]*len(hit_features))
#        feature_scale = np.array([1.,1 ,1 ,np.pi / n_phi_sectors, 1.])
        df = df_all
        if args.muononly: df = df_muon
        df_muon_vps = []
        for entry_all, new_df_all in df.groupby(level=0):
            # Define 'adjacent' layers for each new event
            l = list(set(new_df_all['vh_layer']))
            l_plusZ = np.array(list(filter((0.0).__lt__,l)))
            l_minusZ =np.array(list(filter((0.0).__gt__,l)))
            layer_pairs_plus  = np.stack([l_plusZ[:-1], l_plusZ[1:]], axis=1)
            layer_pairs_plus_one = np.stack([l_plusZ[:-2], l_plusZ[2:]], axis=1)
            layer_pairs_plus_two = np.stack([l_plusZ[:-3], l_plusZ[3:]], axis=1)
            layer_pairs_minus  = np.stack([l_minusZ[1:], l_minusZ[:-1]], axis=1)
            layer_pairs_minus_one  = np.stack([l_minusZ[2:], l_minusZ[:-2]], axis=1)
            layer_pairs_minus_two  = np.stack([l_minusZ[3:], l_minusZ[:-3]], axis=1)

           # layer_pairs = np.concatenate((layer_pairs_plus,layer_pairs_minus,layer_pairs_minus_one, layer_pairs_minus_two,layer_pairs_plus_one,layer_pairs_plus_two),axis=0)
           # layer_pairs = np.concatenate((layer_pairs_plus,layer_pairs_minus,layer_pairs_minus_one,layer_pairs_plus_one),axis=0)
            layer_pairs = np.concatenate((layer_pairs_plus,layer_pairs_minus),axis=0)

            new_df_all = new_df_all.reset_index()
            new_df_all['subentry'] = new_df_all.index
            graph = [construct_graph(new_df_all, layer_pairs=layer_pairs,
                                     feature_names=hit_features,
                                     feature_scale=feature_scale)]
            graphs.append(graph)
            df_muon_vps.append(df_muon_vp.iloc[int(new_df_all.iloc[0]['event_id'])-int(args.start)])
        # Write outputs
        graphs = [g for gs in graphs for g in gs]
        if args.output_dir:
           os.system('mkdir -p %s'%args.output_dir)
           logging.info('Writing outputs to ' + args.output_dir)
        # Write out the graphs
        filenames = [os.path.join(args.output_dir, 'graph_'+file_prefix_muon.split('/')[-1]+'_%06i' % i)
                     for i in range(len(graphs))]
        save_graphs(graphs,df_muon_vps,filenames)
        plotgraphs(args.output_dir)
    return graphs

if __name__ == '__main__':
    main()
