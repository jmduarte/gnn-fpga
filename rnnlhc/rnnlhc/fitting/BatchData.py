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
    def __init__(self):
        return
    #Accepts the batch size as an argument and returns 
    #batch sized data for a specific trajectory length
    def sample_batch(self,rand_int=None,test_idx=None,batch_size=20):
        if rand_int is None:
            #Choose a key from keys randomly
            rand_int = np.random.randint(12,20)
        #Choose indices randomly 
        if test_idx is None:
            idx = np.random.randint(0,len(self.data[str(rand_int)]),batch_size)
        else:
            idx = np.zeros(batch_size)
            counter = 0
            while counter < batch_size:
                tmp = np.random.randint(0,len(self.data[str(rand_int)]))
                if tmp not in test_idx:
                    idx[counter] = tmp
                    counter += 1
        #Init zeros
        data = np.zeros((batch_size,rand_int,self.dims))
        filtered_data = np.zeros((batch_size,rand_int,self.dims))
        for ii in range(batch_size):
            data[ii,:,:] = np.array(self.data[str(rand_int)][idx[ii]])
            filtered_data[ii,:,:] = np.array(self.filtered_data[str(rand_int)][idx[ii]])
        #Return as numpy array

        return data,filtered_data,rand_int,idx

class BatchJsonData(BatchData):
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

class BatchNpyData(BatchData):
    def __init__(self,npy_data):
        #Sorts all the data based on the number of hits
        npy_data = sorted(npy_data, key= lambda x: x[-1])
        len_data = len(npy_data)
        X={}
        hit_len = '0'
        kk = 0
        for ii in np.arange(len_data):
            if hit_len != str(npy_data[ii][-1]):
                hit_len = str(npy_data[ii][-1])
                X[hit_len] = {}
                kk = 0
            if np.abs(npy_data[ii][0]) < 1:
                x = npy_data[ii][2]
                y = npy_data[ii][3]
                z = npy_data[ii][4]
                X[hit_len][kk] = np.vstack((x,y,z)).T
                kk += 1
        self.data = X
        self.dims = 3
        return

class BatchNpyData2(BatchData):
    def __init__(self,npy_data):
        #Sorts all the data based on the number of hits
        npy_data = sorted(npy_data, key= lambda x: x[5])
        len_data = len(npy_data)
        X={}
        X_hat={}
        hit_len = '0'
        kk = 0
        for ii in np.arange(len_data):
            if hit_len != str(npy_data[ii][5]):
                hit_len = str(npy_data[ii][5])
                X[hit_len] = {}
                X_hat[hit_len] = {}
                kk = 0
            if npy_data[ii][3] > .8 and npy_data[ii][3] < 2.3:
                x = npy_data[ii][6]
                y = npy_data[ii][7]
                z = npy_data[ii][8]
                x_hat = npy_data[ii][9]
                y_hat = npy_data[ii][10]
                z_hat = npy_data[ii][12]
                X[hit_len][kk] = np.vstack((x,y,z)).T
                X_hat[hit_len][kk] = np.vstack((x_hat,y_hat,z_hat)).T
                kk += 1
        self.data = X
        self.filtered_data = X_hat
        self.dims = 3
        return
