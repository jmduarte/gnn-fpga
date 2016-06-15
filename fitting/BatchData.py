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
            

        #identify the various sequence lengths
        #####HARDCODE######
        hist_vals, bin_edge = np.histogram(layers,np.arange(11,23))
        self.hist_vals = hist_vals
        self.bin_edge = bin_edge
        #Dictionary empty init
        X = {}
        #For each layer length
        for ii in np.arange(11,23):
            #Find the samples that have this length
            idx = np.where(layers==ii)
            X[str(ii)] = {}
            #Store them in a dictionary
            for jj in range(len(idx)):
                X[str[ii][jj] = json_data['xAOD::Type::TrackParticle']['InDetTrackParticles']['Trk '+ str(idx[jj])]['pos']

        self.data = X
        return
