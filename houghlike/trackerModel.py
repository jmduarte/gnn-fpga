import numpy as np
import math
import random
import h5py
import sys
import optparse
import os

class trackerModel(object):
    def __init__( self, **args ):
        self.target_rank = False #  self.max_find_track will be set
        self.target_curv = False
        self.target_3par = False
        for k,v in args.items():
            setattr(self,k,v)
        self.min_track_radius = self.radius_layer * self.nlayers
        self.max_track_angle = math.atan( (self.lenght_layer / 2.) / (self.radius_layer* self.nlayers) )
        self.structure = np.zeros( (self.nlayers,2) ) ## the shape of the layers
        ##self.layer_data = []
        self.layer_input = []
        self.layer_target = []
        self.full_size = 0
        for layer in range(self.nlayers):
            n_along = int(self.lenght_layer / self.cell_size)
            n_around = int((2*math.pi*(layer+1)*self.radius_layer) / self.cell_size)
            self.structure[layer] = [n_around,n_along]
            ##self.layer_data.append( np.zeros( (n_around, n_along, 2 ) ) ) #input and target
            self.layer_input.append( np.zeros( (n_around, n_along, 1 ) ) ) #input and target
            if self.target_3par:
                self.layer_target.append( np.zeros( (n_around, n_along, 3 ) ) )
            else:
                self.layer_target.append( np.zeros( (n_around, n_along, 1 ) ) )
            ##print "shape of layer",layer,self.layer_data[-1].shape
            print "shape of layer input",layer,self.layer_input[-1].shape
            print "shape of layer target",layer,self.layer_target[-1].shape
            self.full_size += n_along*n_around
        self.filecount=0
    
    def make_full_dataset(self, n_events):
        m = int(200000000. / self.full_size)
        self.filecount = 0
        while n_events>0:
            n_events -= m
            self.make_subset( m )
            print n_events,"to go"
            
    def make_subset(self, n_events):
        while os.path.isfile( 'model_%d.h5'%self.filecount):
            self.filecount += 1
        s = h5py.File('model_%d.h5'%self.filecount,'w')
        
        s['structure'] = self.structure
        full_data = s.create_dataset('input', (n_events,self.full_size) )
        full_target = s.create_dataset('target', (n_events,self.full_size) )
        full_data.attrs.create('cell_size', self.cell_size)
        full_data.attrs.create('radius', self.radius_layer)
        full_data.attrs.create('length', self.lenght_layer)
        full_data.attrs.create('min_tracks',self.min_n_tracks)
        full_data.attrs.create('max_tracks',self.max_n_tracks)
        full_data.attrs.create('n_layers',self.nlayers)
        for i,l in enumerate( self.layer_data ):
            full_data.attrs.create('layer_%d'%i, l.shape)
        for event,(d,t) in enumerate(self.dataset(n_events)):
            full_data[event,:] = d
            full_target[event,:] = t
        
    def generator(self, n_events):
        while 1:
            data = np.zeros((n_events,self.full_size))
            if self.target_rank:
                target = np.zeros((n_events,self.full_size,self.max_find_track+1))
            elif self.target_3par:
                target = np.zeros((n_events,self.full_size,3))
            else:
                target = np.zeros((n_events,self.full_size))
            for event,(d,t) in enumerate(self.dataset(n_events)):
                data[event,:] = d
                target[event,:] = t
            yield data,target
        
    def dataset(self, N=None):
        if N:
            for i in range(N):
                yield self.make_event()
        else:
            while True:
                yield self.make_event()
    
    def make_event(self,targettype='curv'):
        n_tracks = int(random.uniform(self.min_n_tracks, self.max_n_tracks))
        ## re-initialize
        for ilayer in  range(self.nlayers):
            ##self.layer_data[ilayer] = np.zeros( self.layer_data[ilayer].shape )
            self.layer_input[ilayer] = np.zeros( self.layer_input[ilayer].shape )
            self.layer_target[ilayer] = np.zeros( self.layer_target[ilayer].shape )

            
        track_radiuses = []
        for track in range(n_tracks):
            curve = random.uniform( 0.00000001, 1./self.min_track_radius)
            track_radiuses.append( 1. / curve )
        track_radiuses.sort()
        for itrack,track_radius in enumerate(track_radiuses):
            curvature = 1. / (track_radius - self.min_track_radius) 
            rank = itrack+1

            track_angle = random.uniform( - self.max_track_angle, self.max_track_angle )
            track_phi = random.uniform( 0, 2*math.pi)
            track_charge = 1-int(random.uniform(0,2.))
            
            track_3par = [ curvature, track_phi/ 2*math.pi, (track_angle +self.max_track_angle) / (2*self.max_track_angle) ]
                
                            
            if self.target_curv:
                target = curvature
            elif self.target_rank:
                target = rank
            elif self.target_3par:
                target = track_3par
            else:
                raise Execption("no target defined")
            ## find intersections
            for layer in range(self.nlayers):
                z = (layer+1)*self.radius_layer*math.sin( track_angle )
                
                ##iz = int( z/(self.lenght_layer/2.) * self.layer_data[layer].shape[1])
                iz = int( z/(self.lenght_layer/2.) * self.layer_input[layer].shape[1])
                
                local_phi = track_charge * math.acos( (layer+1)*self.radius_layer / (2*track_radius) )
                phi = local_phi + track_phi
                ##put between 0 and 2pi
                while phi>2*math.pi: phi-=2*math.pi
                while phi<0: phi+=2*math.pi
                
                ##iphi = int( phi / (2*math.pi)*self.layer_data[layer].shape[0] )
                iphi = int( phi / (2*math.pi)*self.layer_input[layer].shape[0] )
                
                #print "track",itrack,"layer",layer,"target",target,"radius",track_radius
                #print "phi",phi,"z",z
                #print iphi,iz
                ##self.layer_data[layer][iphi,iz,:] = [1,target] 
                self.layer_input[layer][iphi,iz,:] = 1
                self.layer_target[layer][iphi,iz,:] = target
                #1:pxl is on, target: is the target value
                
                
        #data = np.concatenate(tuple([np.ravel(self.layer_data[layer][...,0]) for layer in range(self.nlayers)]))
        #target = np.concatenate(tuple([np.ravel(self.layer_data[layer][...,1]) for layer in range(self.nlayers)]))
        data = np.concatenate(tuple([np.ravel(self.layer_input[layer]) for layer in range(self.nlayers)]))
        target = np.concatenate(tuple([np.reshape(self.layer_target[layer], (self.layer_target[layer].shape[0]*self.layer_target[layer].shape[1],3)) for layer in range(self.nlayers)]))
        
        if self.target_rank:
            new_target = np.zeros( (target.shape[0], self.max_find_track+1 ))
            for itrack in range(self.max_find_track):
                new_target[np.where(target==itrack+1)[0],itrack] = 1
            new_target[np.where(target>self.max_find_track)[0],self.max_find_track] = 1
            return (data, new_target)
        return (data, target)