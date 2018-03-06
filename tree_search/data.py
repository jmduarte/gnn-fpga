"""
Helper code for parsing and using the ACTS data.
The code in this module depends on pandas.
"""

from __future__ import print_function

import ast
import multiprocessing as mp

import pandas as pd
import numpy as np

def load_data_events(file_name, columns, start_evtid=0):
    """
    Load data from file into a pandas dataframe.
    
    Uses python's ast parser to extract the nested list structures.
    This implementation assumes there is no event ID saved in the file
    and that it must detect events based on the presence of blank lines.
    """
    dfs = []
    print('Loading', file_name)
    with open(file_name) as f:
        event_lines = []
        # Loop over lines in the file
        for line in f:
            # Add to current event
            if line.strip() and line[0] != '#':
                event_lines.append(ast.literal_eval(line))
            
            # Finalize a complete event
            elif len(event_lines) > 0:
                evtid = len(dfs) + start_evtid
                df = pd.DataFrame(event_lines)
                df.columns = columns
                df['evtid'] = evtid
                dfs.append(df)
                event_lines = []
        # Verify there are no leftovers (otherwise fix this code)
        assert len(event_lines) == 0
    
    # Concatenate the events together into one DataFrame
    return pd.concat(dfs, ignore_index=True)

def process_hits_data(df, copy_keys=['evtid', 'barcode', 'volid', 'layid']):
    """Split columns and calculate some derived variables"""
    x = df.gpos.apply(lambda pos: pos[0])
    y = df.gpos.apply(lambda pos: pos[1])
    z = df.gpos.apply(lambda pos: pos[2])
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return df[copy_keys].assign(z=z.astype(np.float32), r=r.astype(np.float32),
                                phi=phi.astype(np.float32))

def read_worker(hits_file):
    hits_columns = ['hitid', 'barcode', 'volid', 'layid', 'lpos',
                    'lerr', 'gpos', 'chans', 'dir', 'direrr']
    return process_hits_data(load_data_events(hits_file, columns=hits_columns))

def process_files(hits_files, num_workers, concat=True):
    """Load and process a set of hits files with MP"""
    pool = mp.Pool(processes=num_workers)
    hits = pool.map(read_worker, hits_files)
    pool.close()
    pool.join()
    # Fix the evtid to be consecutive
    for i in range(1, len(hits)):
        hits[i].evtid += hits[i-1].evtid.iloc[-1] + 1
    if concat:
        return pd.concat(hits, ignore_index=True)
    else:
        return hits
