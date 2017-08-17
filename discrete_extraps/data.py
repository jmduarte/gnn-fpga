"""
Helper code for parsing and using the ACTS data.
The code in this module depends on pandas.
"""

from __future__ import print_function

import ast
import pandas as pd

def load_data_events(file_name, columns, start_evtid=0, print_freq=1e6):
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
                if (evtid % print_freq) == 0:
                    print('Finished event', evtid)
                df = pd.DataFrame(event_lines)
                df.columns = columns
                df['evtid'] = evtid
                dfs.append(df)
                event_lines = []
        # Verify there are no leftovers (otherwise fix this code)
        assert len(event_lines) == 0
    
    # Concatenate the events together into one DataFrame
    return pd.concat(dfs, ignore_index=True)