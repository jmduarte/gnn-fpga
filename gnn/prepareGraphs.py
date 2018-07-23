#!/usr/bin/env python

"""
This script is used to construct the graph samples for
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

from acts import process_hits_files
from graph import construct_graphs, save_graphs


def parse_args():
    parser = argparse.ArgumentParser('prepareGraphs.py')
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/cms-sc17/graphNN/prod_mu10_pt1000_2017_07_29/')
            #default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu10_pt1000_2017_07_29/')
            #default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu200_pt500_2017_07_25')
    add_arg('--output-dir')
    add_arg('--n-files', type=int, default=1)
    add_arg('--n-workers', type=int, default=1)
    add_arg('--n-events', type=int, help='Max events per input file')
    add_arg('--n-tracks', type=int, help='Max tracks per event')
    add_arg('--phi-slope-max', type=float, default=0.001, help='phi slope cut')
    add_arg('--z0-max-inner', type=float, default=200, help='z0 cut, inner layers')
    add_arg('--z0-max-outer', type=float, default=500, help='z0 cut, outer layers')
    add_arg('--quarter-detector', action='store_true',
            help='Build graph just within (z>0 and phi>0)')
    add_arg('--no-missing-hits', action='store_true',
            help='Require no missing hits')
    add_arg('--show-config', action='store_true',
            help='Dump the command line config')
    add_arg('--interactive', action='store_true',
            help='Drop into IPython shell at end of script')
    return parser.parse_args()

def select_hits(hits, quarter_det=False, no_missing_hits=False):
    """
    Selects barrel hits, removes duplicate hits, and re-enumerates
    the volume and layer numbers for convenience.
    If quarter_det parameter is true, then it will only take hits from
    z > 0 and phi > 0, corresponding to one quarter of the detector.
    If no_missing_hits parameter is true, then it will only take tracks
    that hit every layer (10).
    """
    # Select all barrel hits
    vids = [8, 13, 17]
    hits = hits[np.logical_or.reduce([hits.volid == v for v in vids])]
    # Select a subset of the detector
    if quarter_det:
        hits = hits[(hits.z > 0) & (hits.phi > 0)]
    # Re-enumerate the volume and layer numbers for convenience
    volume = pd.Series(-1, index=hits.index, dtype=np.int8)
    vid_groups = hits.groupby('volid')
    for i, v in enumerate(vids):
        volume[vid_groups.get_group(v).index] = i
    # This assumes 4 layers per volume (except last volume)
    layer = (hits.layid / 2 - 1 + volume * 4).astype(np.int8)
    # Select the columns we need
    hits = (hits[['evtid', 'barcode', 'r', 'phi', 'z']]
            .assign(volume=volume, layer=layer))
    if no_missing_hits:
        # Filter tracks that hit every layer                      
        hits = (hits.groupby(['evtid', 'barcode'])
                .filter(lambda x: len(x.layer.unique()) == 10))
    # Remove duplicate hits
    hits = hits.loc[
        hits.groupby(['evtid', 'barcode', 'layer'], as_index=False).r.idxmin()
    ]
    # Require events to have a minimum number of hits
    min_hits = 50
    hits = hits.groupby('evtid').filter(lambda x: x.shape[0] > min_hits)
    return hits

def print_hits_summary(hits):
    """Log some summary info of the hits DataFrame"""
    n_events = hits.evtid.unique().shape[0]
    n_hits = hits.shape[0]
    n_particles = hits[['evtid', 'barcode']].drop_duplicates().shape[0]
    logging.info(('Hits summary: %i events, %i hits, %i particles,' +
                  ' %g particles/event, %g hits/event') %
                 (n_events, n_hits, n_particles,
                  n_particles/n_events, n_hits/n_events))

def print_graphs_summary(sparse_graphs):
    n_events = 0
    n_nodes = []
    n_edges = []
    for sparse_graph in sparse_graphs:
        n_nodes.append(sparse_graph.X.shape[0])
        n_edges.append(sparse_graph.Ri_rows.shape[0])
        n_events+=1
    logging.info(('Graphs summary: %i events, %i edges, %i nodes,' +
                  ' %g edges/event, %g nodes/event') %
                 (n_events, sum(n_edges), sum(n_nodes),
                  sum(n_edges)/n_events, sum(n_nodes)/n_events))

def main():
    """Main program function"""
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Construct layer pairs from adjacent layer numbers 
    layers = np.arange(10)
    layer_pairs = np.stack([layers[:-1], layers[1:]], axis=1)

    # Find the input files
    all_files = os.listdir(args.input_dir)
    hits_files = sorted(f for f in all_files if f.startswith('clusters'))
    hits_files = [os.path.join(args.input_dir, hf)
                  for hf in hits_files[:args.n_files]]

    # Start the worker pool
    with mp.Pool(processes=args.n_workers) as pool:

        # Load the data
        hits = process_hits_files(hits_files, pool)

        # Apply hit selection
        logging.info('Applying hits selections')
        sel_func = partial(select_hits, quarter_det=args.quarter_detector, 
                           no_missing_hits=args.no_missing_hits)
        hits = pool.map(sel_func, hits)

        # Print some summary info
        pool.map(print_hits_summary, hits)

        # Construct graphs of the events
        logging.info('Constructing hit graphs')
        graph_func = partial(construct_graphs,
                             layer_pairs=layer_pairs,
                             phi_slope_max=args.phi_slope_max,
                             z0_max_inner=args.z0_max_inner,
                             z0_max_outer=args.z0_max_outer,
                             max_events=args.n_events,
                             max_tracks=args.n_tracks)

        graphs = pool.map(graph_func, hits)

        pool.map(print_graphs_summary, graphs)

    # Merge across workers into one list of event samples
    graphs = [g for gs in graphs for g in gs]

    # Write outputs
    if args.output_dir:
        logging.info('Writing outputs to ' + args.output_dir)

        # Write out the graphs
        filenames = [os.path.join(args.output_dir, 'event%06i' % i)
                     for i in range(len(graphs))]
        save_graphs(graphs, filenames)

    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
