"""
This module contains the functionality to construct and work with hit graphs.
"""

import logging

from collections import namedtuple

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, find
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


# Global feature details
#feature_names = ['r', 'phi', 'z']
#feature_scale = np.array([1000., np.pi, 1000.])

# Graph is a namedtuple of (X, Ri, Ro, y) for convenience
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','pt','eta'])
# Sparse graph uses the indices for the Ri, Ro matrices
SparseGraph = namedtuple('SparseGraph',['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y','pt','eta'])

def make_sparse_graph(X, Ri, Ro, y):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y)

def graph_from_sparse(sparse_graph, dtype=np.uint8):
    n_nodes = sparse_graph.X.shape[0]
    n_edges = sparse_graph.Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[sparse_graph.Ri_rows, sparse_graph.Ri_cols] = 1
    Ro[sparse_graph.Ro_rows, sparse_graph.Ro_cols] = 1
    return Graph(sparse_graph.X, Ri, Ro, sparse_graph.y, sparse_graph.pt, sparse_graph.eta)

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def select_segments(hits1, hits2, phi_slope_max=10e30, phi_slope_mid_max=10e30, phi_slope_outer_max=10e30, z0_max=10e30):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['event_id', 'vh_layer', 'vh_sim_r', 'vh_sim_phi', 'vh_sim_z']
#    hits1.reset_index(drop=True, inplace=True)
#    hits2.reset_index(drop=True, inplace=True)
    
#    print('HITS1:', hits1)
    #hits1.reset_index(drop=True, inplace=True)
#    print('HITS1 after reset index:', hits1)  
#    print('HITS2 keys:', hits2)
    #print(hits1.reset_index(drop=True))
    hit_pairs = hits1.merge(
        hits2, on="entry",suffixes=('_1', '_2')) 
#    print('HIT PAIRS', hit_pairs)
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.vh_sim_phi_1, hit_pairs.vh_sim_phi_2)
    dz = hit_pairs.vh_sim_z_2 - hit_pairs.vh_sim_z_1
    dr = hit_pairs.vh_sim_r_2 - hit_pairs.vh_sim_r_1
#    print('dr:', dr)
    phi_slope = dphi / dr
    z0 = hit_pairs.vh_sim_z_1 - hit_pairs.vh_sim_r_1 * dz / dr
    #Filter segments according to criteria
    good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)  
    return hit_pairs[['subentry_1', 'subentry_2']][good_seg_mask]

def construct_segments(hits, layer_pairs):
    """
    Construct a list of selected segments from hits DataFrame using
    the specified layer pairs and selection criteria.

    Returns: DataFrame of (index_1, index_2) corresponding to the
    hit indices of the selected segments.
    """
    # Loop over layer pairs and construct segments
    layer_groups = hits.groupby('vh_layer')
    segments = []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try: 
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2) # Not working?
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('SKIPPING empty layer: %s' % e)
            continue
        # Construct the segments
        #print("hits1",hits1)
        #print("hits2",hits2)
        segments.append(select_segments(hits1, hits2))
    #print("segments:",segments)
    # Combine segments from all layer pairs    
    return segments

def construct_graph(hits, layer_pairs,
                    feature_names, feature_scale,
                    max_tracks=None,
                    no_missing_hits=False):
    """Construct one graph (e.g. from one event)"""

    if no_missing_hits:
        hits = (hits.groupby(['isMuon'])
                .filter(lambda x: len(x.layer.unique()) == 12))
    if max_tracks is not None:           
        particle_keys = hits['isMuon'].drop_duplicates().values
        np.random.shuffle(particle_keys)
        sample_keys = particle_keys[0:max_tracks]
        hits = hits[hits['isMuon'].isin(sample_keys)]

    # Construct segments
    segs = construct_segments(hits, layer_pairs)
    try:
        segments = pd.concat(segs)
    except:
        logging.info('SKIPPING empty segments')
        return
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    #evtid = hits.evtid.unique()
    # Prepare the tensors
    #print('HITSX:', hits)
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)
    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    # Get rid of multiindex in hits.index
    #hits.index = hits.index.droplevel(level=0)
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    #print("hits index:",hits.index)
    #print("hit_idx:",hit_idx)
#    print('segments.subentry_1', segments.subentry_1)
#    print('segments.subentry_2', segments.subentry_2)
    seg_start = hit_idx.loc[segments.subentry_1].values
    seg_end = hit_idx.loc[segments.subentry_2].values
    #print("seg_start:",seg_start)
    #print("seg_end:",seg_end)
    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    #print('HITS.INDEX', type(hits.index), hits.index)
#    print('EDGES', np.arange(n_edges), 'RI',  Ri.shape)
    #print('HIT_IDX', type(hit_idx), hit_idx)
    #Ri[seg_end[0], np.arange(n_edges)[0]] = 1
    #Ro[seg_start[0], np.arange(n_edges)[0]] = 1
    #print("n edges:",n_edges)
    #print("segment size:",len(seg_start))
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    #print("Ri shape:", Ri.shape)
    #print("Ri matrix:", Ri)
    #print("Ro matrix:", Ro)
    # Fill the segment labels
    # PROBLEM HERE  
    #pid1 = hits.isMuon.loc[segments.subentry_1.squeeze()].values
    #pid2 = hits.isMuon.loc[segments.subentry_2.squeeze()].values
    
    pid1 = hits.isMuon.loc[segments.subentry_1].values
    pid2 = hits.isMuon.loc[segments.subentry_2].values
    y[:] = [i and j for i, j in zip(pid1, pid2)]
    #print('PID1', hits.isMuon.loc[segments.subentry_1], 'Y:', y)
    # Return a tuple of the results
    #print("X:", X.shape, ", Ri:", Ri.shape, ", Ro:", Ro.shape, ", y:", y.shape)
    return make_sparse_graph(X, Ri, Ro, y), segments  
    #return Graph(X, Ri, Ro, y)

def construct_graphs(hits, layer_pairs,
                     max_events=None, max_tracks=None):
    """
    Construct the full graph representation from the provided hits DataFrame.
    Returns: A list of (X, Ri, Ro, y)
    """
    # Organize hits by event 
    evt_hit_groups = hits.groupby('event_id')
    # Organize hits by event and barcode
    evt_barcode_hit_groups = hits.groupby(['event_id', 'entry'])
    #print(evt_barcode_hit_groups)
    evtids = hits.event_id.unique()
    if max_events is not None:
        evtids = evtids[:max_events]

    # Loop over events and construct graphs
    graphs = []
    for evtid in evtids:
        # Get all the hits for this event
        evt_hits = evt_hit_groups.get_group(evtid)
        if max_tracks is not None:
            particle_keys = evt_hits['entry'].drop_duplicates().values
            np.random.shuffle(particle_keys)
            sample_keys = particle_keys[0:max_tracks]
            evt_hits = evt_hits[evt_hits['entry'].isin(sample_keys)]
            #print('max tracks:', len(sample_keys))
            #print('number of hits:', len(evt_hits))
        graph = construct_graph(evt_hits, layer_pairs,
                                phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max_inner, z0_max_outer)
        graphs.append(graph)

    # Return the results
    return graphs

def save_graph(graph,particle ,filename):
    """Write a single graph to an NPZ file archive"""
    try:
       np.savez(filename, **graph[0]._asdict(),pt = particle.vp_pt,eta = particle.vp_eta)
    except:
       print("empty graph")
       return
    #np.savez(filename, X=graph.X, Ri=graph.Ri, Ro=graph.Ro, y=graph.y)

def save_graphs(graphs, particles,filenames):
    for graph, particle,filename in zip(graphs,particles, filenames):
        save_graph(graph,particle ,filename)

def load_graph(filename, graph_type=Graph):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return graph_type(**dict(f.items()))

def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]

def draw_sample(X, Ri, Ro, y, 
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None): 
    # Select the i/o node features for each segment    
    # Prepare the figure
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))
    cmap = plt.get_cmap(cmap)
    #print("input features:",find(np.rot90(Ri))) 
    #print("output features:", find(np.rot90(Ro)))
    #print('X:',X) 
    #print("X shape:",X.shape)
    feats_o = X[find(np.rot90(Ri))[1]]
    feats_i = X[find(np.rot90(Ro))[1]]  
    #print("feats_o:",feats_o)
    #print("feats_i:",feats_i)
    
    if sim_list is None:    
        # Draw the hits (layer, theta, z)
        ax0.scatter(X[:,2], X[:,1], c='k')
        ax1.scatter(X[:,0], X[:,1], c='k')
    else:        
        #ax0.scatter(X[:,0], X[:,2], c='k')
        #ax1.scatter(X[:,1], X[:,2], c='k')
        ax0.scatter(X[sim_list,0], X[sim_list,2], c='b')
        ax1.scatter(X[sim_list,1], X[sim_list,2], c='b')
    
    # Draw the segments
    for j in range(y.shape[0]):
        if not y[j] and skip_false_edges: continue
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))
        ax0.plot([feats_o[j,2], feats_i[j,2]],
                 [feats_o[j,1],feats_i[j,1]], '-', **seg_args)
        ax1.plot([feats_o[j,0], feats_i[j,0]],
                 [feats_o[j,1],feats_i[j,1]], '-', **seg_args)
        #ax0.plot([feats_o[0,0], feats_i[0,0]],
        #         [feats_o[0,2], feats_i[0,2]], '-', **seg_args)
        #ax1.plot([feats_o[0,1], feats_i[0,1]],
        #         [feats_o[0,2], feats_i[0,2]], '-', **seg_args)
    # Adjust axes
    ax0.set_xlabel('$layer$ [arb]')
    ax0.set_ylabel('$theta$')
    ax1.set_xlabel('$z$ [cm]')
    ax1.set_ylabel('$theta$')
    plt.tight_layout()
    plt.savefig("graph.png")

def draw_sample_withproperties(X, Ri, Ro, y, pt, eta, 
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None,outputname=None,output=None): 
    # Select the i/o node features for each segment    
    # Prepare the figure
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))
    cmap = plt.get_cmap(cmap)
    #print("input features:",find(np.rot90(Ri))) 
    #print("output features:", find(np.rot90(Ro)))
    #print("X shape:",X.shape)
    feats_o = X[find(np.rot90(Ri))[1]]
    feats_i = X[find(np.rot90(Ro))[1]]  
    #print("feats_o:",feats_o)
    #print("feats_i:",feats_i)
    plt.title('Muon properties Pt: %f, Eta: %f'%(pt,eta) )    
    if sim_list is None:    
        # Draw the hits (layer, theta, z)
        ax0.scatter(X[:,10], X[:,1], c='k')
        ax1.scatter(X[:,0], X[:,10], c='k')
    else:        
        ax0.scatter(X[sim_list,0], X[sim_list,2], c='b')
        ax1.scatter(X[sim_list,1], X[sim_list,2], c='b')
    
    # Draw the segments
    for j in range(y.shape[0]):
        if not y[j] and skip_false_edges: continue
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))
        ax0.plot([feats_o[j,10], feats_i[j,10]],
                 [feats_o[j,1],feats_i[j,1]], '-', **seg_args)
        ax1.plot([feats_o[j,0], feats_i[j,0]],
                 [feats_o[j,10],feats_i[j,10]], '-', **seg_args)
    # Adjust axes
    ax0.set_xlabel('$layer$ [arb]')
    ax0.set_ylabel('$theta$')
    ax1.set_xlabel('$z$ [cm]')
    ax1.set_ylabel('$layer$ [arb]')
    plt.tight_layout()
    plt.savefig("myplots/%s/graph_%s.png"%(outputname,output) )