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

