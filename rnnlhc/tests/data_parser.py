'''
Script to test loading data from Json file
and using the Data object parser

Mayur Mudigonda
'''

import numpy as np
from rnnlhc.fitting.utilities import parse_data
from rnnlhc.fitting import BatchData

json_data = parse_data('../data/EventDump_10Ktracks.json')

BD = BatchData.BatchData(json_data=json_data)

import IPython; IPython.embed()

BD.sample_batch(10)
