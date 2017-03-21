#!/usr/bin/python
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

#import theano
#theano.config.device = 'gpu'


import numpy
import numpy as np
import scipy

numpy.random.seed(1337)  # for reproducibility

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Optimizer
from keras.utils import np_utils

#from keras.functions import merge
from keras.layers import Input, Dense, Merge, merge, Lambda, MaxoutDense, BatchNormalization,LSTM,TimeDistributed
from keras.models import Model
import theano
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU,ParametricSoftplus,PReLU



from keras.engine.topology import Layer


import math

import matplotlib.pyplot as plt


from keras.backend.common import _FLOATX, _EPSILON, _IMAGE_DIM_ORDERING

#from ROOT import gROOT, TH1D

from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType
from theano.tensor import basic as tensor

#from rnnlhc.fitting import BatchData
import json


'''
Class that takes the whole JSON data dump
and sorts them into dictionaries of varied lengths
of trajectories.
After this is done during init, when we request a batch
of data, we can choose an arbitrary sequence length
and the appropriate batch size
'''
import numpy as np

class BatchData:
    def __init__(self,json_data):
        num_samples = len(json_data['xAOD::Type::TrackParticle']['InDetTrackParticles'])
        data = []
        layers = []
        for ii in np.arange(num_samples):
            tmp = json_data['xAOD::Type::TrackParticle']['InDetTrackParticles']['Trk '+ str(ii)]['pos']
            data.append(tmp)
            layers.append(len(tmp))
            

        #identify the various sequence lengths
        #####HARDCODE######
        hist_vals, bin_edge = np.histogram(layers,np.arange(11,23)) # might go lower than 11
        self.hist_vals = hist_vals
        self.bin_edge = bin_edge
        #Dictionary empty init
        X = {}
        #For each layer length
        ######HARDCODE#####
        for totl,ii in enumerate(np.arange(12,20)):
            #Find the samples that have this length
            idx = np.where(layers==ii)
            idx = np.array(idx)
            X[str(ii)] = {}
            #Store them in a dictionary
            for jj in range(idx.shape[1]):
                X[str(ii)][jj] = json_data['xAOD::Type::TrackParticle']['InDetTrackParticles']['Trk '+ str(idx[0][jj])]['pos']

        self.data = X
        self.total_keys = totl
        self.dims = 3
        return

    #Accepts the batch size as an argument and returns 
    #batch sized data for a specific trajectory length
    def sample_batch(self,rand_int=None,batch_size=20):
        if rand_int is None:
            #Choose a key from keys randomly
            rand_int = np.random.randint(12,20)
        #Choose indices randomly 
        idx = np.random.randint(0,len(self.data[str(rand_int)]),batch_size)
        #Init zeros
        data = np.zeros((batch_size,rand_int,self.dims))
        for ii in range(batch_size):
            data[ii,:,:] = np.array(self.data[str(rand_int)][idx[ii]])
        #Return as numpy array

        return data,rand_int


iter_size=8192
batch_size=128
niter = 3000
nval = 1024

ndim = 3
nhidden = 10
timesteps = 12

jsoninput = '../data/EventDump_10Ktracks.json'
json_data = open(jsoninput,'r').read()
parsed_json_data = json.loads(json_data)
BD = BatchData(parsed_json_data)

valdata,rand_int = BD.sample_batch(timesteps,nval)
valindata = valdata[:,0:timesteps-1]
valtarget = valdata[:,1:timesteps]


inputs = Input(shape=(timesteps-1,ndim))
outputs = inputs
outputs = LSTM(output_dim=nhidden,return_sequences=True,init='glorot_uniform')(outputs)
outputs = TimeDistributed(Dense(nhidden,activation='relu',init='glorot_uniform'))(outputs)
outputs = TimeDistributed(Dense(ndim,activation='linear',init='glorot_uniform'))(outputs)
model = Model(input=inputs,output=outputs)
model.summary()

model.compile(loss='mse',
            optimizer='Nadam',
            metrics=['accuracy']
            )

for ibatch in range(niter):
  print(ibatch)
  
  data,rand_int = BD.sample_batch(timesteps,iter_size)
  indata = data[:,0:timesteps-1]
  target = data[:,1:timesteps]
  
  
  history = model.fit(indata, target,
                      batch_size=batch_size, nb_epoch=1,
                      verbose=1, validation_data=(valindata, valtarget))  
  
ntest = 1000
data,rand_int = BD.sample_batch(timesteps,ntest)
indata = data[:,0:timesteps-1]
target = data[:,1:timesteps]

outdata = model.predict(indata,batch_size=batch_size)

diff = outdata - target

plt.figure()
plt.hist(diff[:,0], bins=100,range=[-100.,100.])

plt.figure()
plt.hist(diff[:,1], bins=100,range=[-100.,100.])

plt.figure()
plt.hist(diff[:,2], bins=100,range=[-100.,100.])

plt.show()


  
