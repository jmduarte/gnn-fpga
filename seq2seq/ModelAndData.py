import os
import glob
import pickle
import h5py
import time
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict

if True:#False:#True:
    ## CPU only
    os.environ['KERAS_BACKEND']='tensorflow'
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES']=""
    print os.environ['CUDA_VISIBLE_DEVICES']
    import tensorflow as tf        
else:
    os.environ['KERAS_BACKEND']='theano'

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Merge, Dense, Dropout, merge, Reshape, Dropout
from keras.layers import LSTM, Lambda, TimeDistributed, Activation, RepeatVector, Permute, Masking
from keras.layers.advanced_activations import ELU

class DatasetDealer:
    def __init__(self, **params):
        self.filename = params['filename'] if 'filename' in params else 'public_train.csv'
        self.label = params['label'] if 'label' in params else ''
        self.tf = None
        self.all_phi = [ 9802,  21363,  38956,  53533,  68110,  50894,  70624,  95756, 125664]
        
    def create_bank(self, parse=1000):
        if type(self.tf) == pd.core.frame.DataFrame:
            return
        print "Retrieving a",parse,"track bank. This can take a while"
        df = pd.read_csv(self.filename)        
        max_count=0
        index=0
        self.tf = pd.DataFrame(columns=('track_id', 'layer', 'iphi','x','y'))
        for count,ievent in enumerate(list(set(df['event_id'].values))[:parse]):
            sub_data = df.loc[df['event_id']==ievent]
            augment = defaultdict(list)
            for x,y,layer,iphi,aclass in zip(sub_data['x'].values,sub_data['y'].values,sub_data['layer'].values,sub_data['iphi'].values,sub_data['cluster_id'].values ):
                augment[max_count+aclass].append([int(max_count+aclass),int(layer),int(iphi),x,y])
                
            max_count+= max(sub_data['cluster_id'].values)+1
            for trackid in augment:
                for point in sorted(augment[trackid], key = lambda i:i[1]):
                    self.tf.loc[index] = point
                    index+=1
        #self.tf.to_csv('track_bank.csv', index=False)
        
    def generate(self, N=100, mu=10 ,offset=0):
        self.create_bank()
        print "Generating",N,"events with an average",mu,"tracks. This can take a while."
        track_index = set(self.tf['track_id'].values)
        n_tracks = np.random.poisson(mu, size=(N,))
        data = pd.DataFrame(columns = ('event_id','cluster_id','layer', 'iphi','x','y'))
        master_index=0
        for ievent,n_track in enumerate(n_tracks):
            #print n_track
            phis = np.random.random(size=(n_track,))*np.pi*2
            ## pick at random n_track in the bank
            for i_track,track_id in enumerate( np.random.choice(list(track_index), n_track) ):
                ## the angle by which the track is going to be rotated
                phi = phis[i_track]
                track_data = self.tf.loc[self.tf['track_id'] == track_id].values #[0:'track_id', 1:'layer', 2:'iphi',3:'x',4:'y']
                ## modify iphi accordingly
                for i, layer in enumerate(track_data[:,1]):
                    track_data[i,2] = int(track_data[i,2] + self.all_phi[int(layer)]*phi/(np.pi*2))%self.all_phi[int(layer)]
                x = track_data[:,3]
                y = track_data[:,4]
                iphi = track_data[:,2]
                ## rotate the initial hit coordinates in x,y (N.B. this is not used eventually in the model. so we could skip it for speed-up
                c, s = np.cos(phi), np.sin(phi)
                R = np.matrix([[c, -s], [s, c]])                
                coord = np.matrix([x,y])
                coord = np.dot(R,coord)
                        
                # rotated track
                track_data[:,3] = np.ravel(coord[0,...])
                track_data[:,4] = np.ravel(coord[1,...])
                for entry in range(track_data.shape[0]):
                    l = np.zeros((6,))
                    l[:2] = [offset+ievent,i_track]
                    l[2:] = track_data[entry,1:]
                    data.loc[master_index] = l
                    master_index+=1

        ##cast the types of the table
        data['event_id'] = data['event_id'].astype(int)
        data['layer'] = data['layer'].astype(int)
        data['cluster_id'] = data['cluster_id'].astype(int)
        data['iphi'] = data['iphi'].astype(int)
        return data
                        
    def create_dataset(self,
                       N=10000,
                       mu=10):
        self.base = "dataset%s_mu%d_"%(self.label,mu)
        per_file=1000
        for i_batch in itertools.count():
            out_file = "%s%d.csv"%(self.base,i_batch+1)
            offset = i_batch*per_file
            if offset >= N: break
            if os.path.isfile( out_file ):
                print out_file,"exists_already"
                continue
            print "creating data for",out_file,"this might take a while"
            df = self.generate(N=per_file, mu=mu,
                                offset = offset)
            df.to_csv( out_file, index=False)
            a = set(df['event_id'])
            print "Created range of events",min(a),max(a)
        return self.base
                        
class ModelDealer:
    def __init__(self, **params):
        self.base = 'dataset_mu10_' if not 'base' in params else params['base']
        self.all_phi= [ 9802,  21363,  38956,  53533,  68110,  50894,  70624,  95756, 125664]
        self.radiuses = [38, 84, 154, 212, 270, 405, 562, 762, 1000 ]
        self.n_layers= 4 if not 'n_layers' in params else params['n_layers']
        self.max_phi= self.all_phi[:self.n_layers] 
        self.downcast=500 if not 'downcast' in params else params['downcast']

        self.hits_per_layer = 20
        self.tracks_candidates = 5
        #n_layers = len(max_phi)
        self.masking_char = -1.
        self.model_label = "all_%dL_%dH_%dC"%(self.n_layers,
                                              self.hits_per_layer,
                                              self.tracks_candidates )
        if self.downcast:
            self.downcast_phi = [min(self.downcast,i) for i in self.max_phi] ## downcast only, no upcasting
            self.model_label += '_DC%s'%self.downcast
        else:
            self.downcast_phi = self.max_phi
                
        print "Ready for model",self.model_label
        self.round_count = 0

    
    def get_event(self, df, ievent , norm_out= False):
        #print ievent
        sub_data = df.loc[df['event_id']==ievent]
        data = np.full((self.n_layers, self.hits_per_layer), self.masking_char)
        target = np.full( (self.tracks_candidates, self.n_layers), self.masking_char)
        truth = defaultdict(list)
        #large_target = []
        #very_large_target = []
        for layer,w in enumerate(self.downcast_phi):
            for track in range(self.tracks_candidates):
                target[track,layer] = w
                #large_target.append( np.full((1,1), w).astype(int) ) ## the default is the layer len, meaning no hit category
                #very_large_target.append( np.zeros((1,w)) )
                
        per_layer = np.zeros( (self.n_layers,), dtype=int)
        first_phis = [None]*self.tracks_candidates        
        for layer,iphi,aclass in zip(sub_data['layer'].values,sub_data['iphi'].values,sub_data['cluster_id'].values ):
            norm_phi = iphi / float(self.all_phi[layer])

            if layer>= self.n_layers: continue ## more layers than the model can handle
                    
            truth[aclass].append( [layer,iphi] )
            
            if per_layer[layer]>= self.hits_per_layer: continue ## more entries per layer than the mode can handle
            ## that the main input/output value
            #
            data[layer, per_layer[layer] ] = norm_phi
            #trackid[layer, per_layer[layer] ] = aclass
            per_layer[layer]+=1
                    
            downcast_iphi = int(norm_phi*self.downcast_phi[layer]) if self.downcast else iphi
                    
            if aclass>=self.tracks_candidates: continue ## more track than the model can handle
            if first_phis[aclass] ==None: first_phis[aclass] = norm_phi
            
            if norm_out:
                target[aclass,layer] = norm_phi ## for a regression type of thing
            else:
                target[aclass,layer] = downcast_iphi ## for sparse categorical cross entropy style
                #lin_index = layer*self.tracks_candidates + aclass
                #large_target[lin_index][0,0] = downcast_iphi

        ## do some ordering
        for layer,w in enumerate(self.downcast_phi):
            ## order in increasing per layer
            with_data = np.where( data[layer,:] != self.masking_char)
            in_sort = np.argsort( data[layer,:][with_data] )
            data[layer,:len(in_sort)] = data[layer, in_sort ]
        ## order track tragets by phi of first hit
        in_sort = np.argsort( first_phis )
        target = target[in_sort,:]
            
        data = np.expand_dims(data, 0)
        if not norm_out:
            target = target.astype(int)
                            
        vtruth = [] ## a list of lists
        for i,thits in truth.items(): vtruth.append( thits )
        flat_target = list([ target[h,layer].reshape(1,1) for layer,h in itertools.product( range(self.n_layers), range(self.tracks_candidates)) ])
        return data, flat_target, vtruth
        #return data, large_target, vtruth

    def make_dataset(self,df,limit=None):
        n_events = len(set(df['event_id'].values)) if not limit else limit
        print n_events,"to be collected"
        
        data =None
        target = None
        truth = []
        
        start = time.mktime(time.gmtime())
        for count,ievent in enumerate(list(set(df['event_id'].values))[:n_events]):
            d,t,tt = self.get_event(df, ievent)
            
            truth.append( tt ) # list of list of list [event][track][hits]:[layer,iphi]
            if data==None or target == None:
                data = d
                target = t
            else:
                data= np.concatenate((data,d))
                if type( t) == list:
                    for lin_index in range(len( target )):
                        target[lin_index] = np.concatenate((target[lin_index],t[lin_index]))
                else:
                    target = np.concatenate((target,t))
            if (count+1)%1000 == 0:
                now = time.mktime(time.gmtime())
                print count,"so far",now-start,"[s]"
                start = now
        return data,target, truth

    def save_dataset(self,data,target,truth, fname):
        print "Creating dataset in ",fname
        f = h5py.File(fname,'w')
        f['data'] = data
        f['target'] = target
        f.close()
        p = open(fname+'.pkl','w')
        pickle.dump( truth, p )
        p.close()
        
    def get_frame(self,f):
        df = pd.read_csv(f)
        df['event_id'] = df['event_id'].astype(int)
        df['layer'] = df['layer'].astype(int)
        df['iphi'] = df['iphi'].astype(int)
        df['cluster_id'] = df['cluster_id'].astype(int)
        return df

    def convert_file(self,f,fh5,force=False):
        if os.path.isfile(fh5) and not force:
            print fh5,"exists already"
            return
        print "converting",f
        
        df =self.get_frame(f)
        data, target, truth = self.make_dataset(df)
        self.save_dataset( data, target, truth, fh5 )
        
        
    def convert_files(self,specific=None,
                      force=False):
        pattern = '%s%s.csv'%(self.base , '*' if not specific else specific)
        extra = self.model_label
        converted = []
        for f in glob.glob(pattern):
            fh5 = '%s%s.h5'%(f,extra)
            self.convert_file(f,fh5,force=force)
            converted.append( fh5 )
        return converted

    def list_dataset(self):
        pattern = '%s*%s.h5'%(self.base,self.model_label)
        return sorted(glob.glob(pattern))
    
    def read_dataset(self,
                     specific=None,
                     andtruth=False):
        pattern = '%s*%s.h5'%(self.base,self.model_label) if not specific else specific
        
        print pattern
        data = None
        target = None
        truth=[]
        for fname in glob.glob(pattern):
            print fname
            f = h5py.File(fname)
            d = f['data'].value
            t = f['target'].value
            print d[0][0][:3]
            print t[0][0][:3]
            #print data.shape if data!=None else "NA",d.shape
            #print target.shape if target!=None else "NA",t.shape
            if data==None or target == None:
                data = d
                target = t
            else:
                data= np.concatenate((data,d))
                if type( t) == list:
                    for lin_index in range(len( target )):
                        target[lin_index] = np.concatenate((target[lin_index],t[lin_index]))
                else:
                    #try:
                    #    target = np.concatenate((target,t))
                    #except:
                    target = np.concatenate((target, t), axis=1)
            f.close()
            if andtruth:
                try:
                    p=open(fname+'.pkl')
                    truth.extend( pickle.load(p) )
                    p.close()
                except:
                    pass
            target = [target[i,...] for i in range(target.shape[0])]
        return data,target, truth

    def make_model(self):
        self.round_count = 0
        inputs = []
        lstms = []
        main = Input( (self.n_layers,self.hits_per_layer) , name='phi_layers')
        self.hidden_size_1 = 15
        self.hidden_size_2 = 17

        for l in range(self.n_layers):
            ## get a vector for each layer
            i = Lambda(lambda x: x[:,l,:], name = 'split_%d'%l, output_shape = (self.hits_per_layer,))(main)
            ## get the vector into a sequence of single float
            i = Reshape((self.hits_per_layer,1), name='sequence_layer_%d'%l)(i)
            ## mask all zeros to the LSTM
            m = Masking(mask_value=self.masking_char, name='masking_layer_%d'%l)(i)
            ## each sequence of phis by layer goes into its own lstm
            lstm = LSTM(self.hidden_size_1,
                        name='encoder_lstm_%d'%l) ( m ) ## should this lstm be shared between layers ?
            lstm = Reshape( (1,self.hidden_size_1), name='lstm_sequence_%d'%l) (lstm)
            inputs.append( i )
            lstms.append( lstm )

        ##stack the output of all lstms of each layer
        #m = merge(lstms, mode='concat', concat_axis=1)
        m = merge(lstms, mode='concat', concat_axis=1, name='core_merge')
    
        m = Flatten()(m)
        m = RepeatVector( self.tracks_candidates )(m)
        ## reshape it as a sequence of vector of hidden_size
        m = Reshape((self.tracks_candidates*self.n_layers,self.hidden_size_1 ))(m)
        ## run it through the LSTM, returning the full sequence
        m = LSTM(self.hidden_size_2, name='core_lstm', return_sequences=True)(m)
        ## picking up the lstm output time n_layers input vector were used
        m = Lambda(lambda x: x[:,::self.n_layers,:], name = 'every_other_%d'%self.n_layers, output_shape = (self.tracks_candidates,self.hidden_size_2))(m)
        
        outputs=[]
        picks = [Lambda(lambda x: x[:,cand,:], name = 'pick_seq_%d'%(cand), output_shape=(self.hidden_size_2,)) for cand in range(self.tracks_candidates)]
        for l,w in enumerate(self.downcast_phi):
            decoder = Dense(w+1, activation='linear', name='decoder_%d'%l)## the +1 is for the "no hit" category
            sub_outputs = []
            for cand in range(self.tracks_candidates):
                #pick= picks[cand](m)
                outputs.append( Activation('softmax')(decoder(picks[cand](m))))
            
        
        self.model = Model( input=main ,output= outputs)
        self.model.summary()

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
        
        return self.model

    def expected_weights(self):
        return 'model_%s_%d_w.h5'%(self.model_label, self.round_count)
    
    def _roll_up(self, data=None, target=None, andTrain=False):
        ##andTrain could be infered from data and target.
        ## will look for the existing weights and train for the next one
        last_weights =None
        base_count = self.round_count
        all_weights = []
        for t in itertools.count():
            self.round_count = base_count+t
            out_weights = self.expected_weights()
            if os.path.isfile( out_weights ):
                last_weights = out_weights
                all_weights.append( out_weights )
                print "\t",out_weights,"found. skipping step"
                continue
            if last_weights:
                print last_weights,"found. Using it"
                self.model.load_weights( last_weights )
            if not andTrain: break
            
            print "Starting training round",self.round_count,time.asctime()
            now = time.mktime(time.gmtime())
            self.model.fit( data, target, batch_size=1000, nb_epoch=100, verbose=0)
            then = time.mktime(time.gmtime())
            print "Trained in",then-now,"[s]"
            self.model.save_weights( out_weights )
            all_weights.append( out_weights )            
            break
        return all_weights
    
    def best_model(self):
        ## load the best existing weights
        return self._roll_up()
        
    def pick_and_train(self, data, target):
        return self._roll_up( data, target, andTrain=True)[-1]
        
    def loop_train(self, data, target,
                   max_round=10,
                   force=False):
        last_weights =None

        base_count = self.round_count
        for t in range(max_round):
            out_weights = self.pick_and_train(data, target)

            ev = self.model.evaluate( data, target, batch_size=1000)
            print "round",t,"/",max_round, ev
            continue

    def qsub_train(self, max_round=10):
        ##drive the training with successive submission to qsub
        return

    def visualize( self,
                   index , data, target, prediction, truths,
                   showCand=True,
                   showTarget=True,
                   showHits=True,
                   showTruth=True,
                   extraTitle=''):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        truth = truths[index]
        ## hits is a 9 long list of a hits_per_layer-sequence of iphi values [0,1] <= [0,layer_size]
        hits = data[index]
        ## tracks is a (track_candidates*9)-long list of category : iphi==position, layer_size == no hit
        tracks_l = [t[index] for t in target]
        tracks = np.zeros((self.tracks_candidates, self.n_layers ))
        for layer in range(self.n_layers):
            for itrack in range(self.tracks_candidates):
                tracks[itrack,layer] = tracks_l[layer*self.tracks_candidates + itrack]
                
        ## candidates_prob is a (track_candidates*9)-long list of layer_size-category probabilities
        candidates_prob = [p[index] for p in prediction]
        ## candidates_cat : cast probability into the argmax to have the category
        candidates_cat = [np.argmax(prob) for prob in candidates_prob]
        
        ## candidates can be compared with tracks directly
        candidates = np.zeros((self.tracks_candidates, self.n_layers ))
        plt.figure( figsize=(10,10))
        plt.subplot(aspect='equal')
        for layer in range(self.n_layers):
            plt.gcf().gca().add_artist( plt.Circle((0, 0), self.radiuses[layer],color='g',  fill=False ) )
            for itrack in range(self.tracks_candidates):
                candidates[itrack,layer] = candidates_cat[layer*self.tracks_candidates + itrack]
                        
        ## now you can compare
        truths=[]
        preds= []
        potentials = []
        def make_range( l ):
            if l:
                make_range.xr = max(make_range.xr,max([abs(i[0])*make_range.factor for i in l]))
                make_range.yr = max(make_range.yr,max([abs(i[1])*make_range.factor for i in l]))
        make_range.xr = make_range.yr = 0
        make_range.factor = 1.2
        
        for itrack,t in enumerate(truth):
            truth_t = []
            for layer,iphi in t:
                if layer>=self.n_layers:continue
                nphi = iphi / float(self.max_phi[layer])
                x = self.radiuses[layer]* np.cos( nphi * np.pi*2.)
                y = self.radiuses[layer]* np.sin( nphi * np.pi*2.)
            truth_t.append( (x,y))
            make_range(truth_t)
            if showTruth:
                plt.plot( [i[0] for i in truth_t], [i[1] for i in truth_t],
                          label='truth %d'%itrack,
                          marker='s',
                          fillstyle='none',
                          color='black'
                )
        all_hits = []
        for layer in range(hits.shape[0]):
            for i in range(hits.shape[1]):
                nphi = hits[layer,i]
                if nphi==self.masking_char: continue
                x = self.radiuses[layer]* np.cos( nphi * np.pi*2.)
                y = self.radiuses[layer]* np.sin( nphi * np.pi*2.)
                all_hits.append( (x,y) )
        make_range(all_hits)
        if showHits:
            plt.plot([i[0] for i in all_hits],
                     [i[1] for i in all_hits],
                     label = 'all hits in data',
                     marker = '*',
                     color='black',
                     linestyle='None'
        )

        for itrack in range(self.tracks_candidates):
            #print "candidate",itrack
            ttarget= []
            pred= []
            potential = []
            for layer in range(self.n_layers):
                ## values are in iphi, downcasted
                dc_iphi_pred = candidates[itrack,layer]
                if dc_iphi_pred == self.downcast_phi[layer]:
                    ## no hit predicted
                    pass
                else:
                    nphi_pred = candidates[itrack,layer] / float(self.downcast_phi[layer])
                    x = self.radiuses[layer]* np.cos( nphi_pred * np.pi*2.)
                    y = self.radiuses[layer]* np.sin( nphi_pred * np.pi*2.)
                    pred.append( (x,y) )
                    
                dc_iphi_true = tracks[itrack,layer]
                if dc_iphi_true == self.downcast_phi[layer]:
                    ## no hit expected
                    pass
                else:
                    nphi_true = tracks[itrack,layer] / float(self.downcast_phi[layer])
                    x = self.radiuses[layer]* np.cos( nphi_true * np.pi*2.)
                    y = self.radiuses[layer]* np.sin( nphi_true * np.pi*2.)
                    ttarget.append( (x,y) )
            c=cm.hsv( (itrack+2)/float(self.tracks_candidates+2))
            make_range( ttarget )
            make_range( pred )
            if showTarget:
                plt.plot( [i[0] for i in ttarget], [i[1] for i in ttarget],
                          label='target %d'%itrack,
                          #marker='d',
                          marker='+',
                          fillstyle='none',
                          linestyle='--',
                          #color='black',
                          c=c,
                )
            if showCand:
                plt.plot( [i[0] for i in pred], [i[1] for i in pred],
                          label='cand %d'%itrack,
                          marker='o',c=c)
        plt.ylim((-make_range.yr,make_range.yr))
        plt.xlim((-make_range.xr,make_range.xr))
        plt.title("Event %d %s"%(index,extraTitle))
        plt.legend(loc=(1,0))
        
        plt.show()
                                                                                
    def test(self, one_file):
        c = self.convert_files(one_file)
        data, target, _ = self.read_dataset(c[0])
        self.make_model()
        self.loop_train( data, target , max_round = 3)
        
if __name__ == '__main__':
    
    D = DatasetDealer()
    D.create_dataset(N=10000)
    
    M = ModelDealer( downcast = 700 , base = D.base)
    M.test(102)
    
    
    #M.make_model()
    #M.convert_files('dataset_mu10_101.csv')
    #data, target, _ = M.read_dataset('dataset_mu10_101.csv%s.h5'%M.model_label)
    #M.loop_train( data, target )
    
