'''
Class that is now a utility class. It does things like
load data, visualize data, plot data and more
Mayur Mudigonda, June 2016
'''
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def parse_data(fname='../data/EventDump_10tracks.json',num_samples=None):
    json_data = open(fname,'r').read()
    parsed_json_data = json.loads(json_data)
    return parsed_json_data

def plot_samples(parsed_data,num_samples=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if num_samples is None:
        #Calculate max number of samples
        #Setting it to 10 for now
        num_samples = parsed_data.shape[0]

    for ii in np.arange(num_samples):
        data = parsed_data['xAOD::Type::TrackParticle']['InDetTrackParticles']['Trk '+ str(ii)]['pos']
        data = np.array(data)
        ax.plot(data[:,0],data[:,1],data[:,2],linestyle='-',linewidth=3.2,label='Track '+str(ii))

    ax.view_init(elev=18,azim=-27)
    ax.hold(True)
    ax.dist=9
    plt.show()
    return

def proj_2d_plot(data,reconstr,savestr='proj_',samples=15):
    for ii in np.arange(samples):
        plt.figure(figsize=(20,20))
        for jj in np.arange(3):
            plt.subplot(int('13'+str(jj+1)))
            pt_d, = plt.plot(data[ii,1:,jj],data[ii,1:,np.mod(jj+1,3)],'r+',label='data')
            plt.hold(True)
            line_d,= plt.plot(data[ii,1:,jj],data[ii,1:,np.mod(jj+1,3)],'r')
            pt_r, = plt.plot(reconstr[ii,:,jj],reconstr[ii,:,np.mod(jj+1,3)],'g*',label='reconstr')
            line_r, = plt.plot(reconstr[ii,:,jj],reconstr[ii,:,np.mod(jj+1,3)],'g')
            plt.legend(handles=[pt_d,pt_r],loc=0)
            plt.locator_params(nbins=10)
            delta = 0.03
            plt.axis([data[ii,1:,jj].min()+ (-delta) ,data[ii,1:,jj].max() + delta ,data[ii,1:,np.mod(jj+1,3)].min() - delta ,data[ii,1:,np.mod(jj+1,3)].max() + delta])
            if jj == 0:
              plt.xlabel('R Phi vs Z')
            elif jj == 1:
              plt.xlabel('Z vs R')
            else:
              plt.xlabel('R vs R Phi')
        plt.savefig('png/'+savestr+str(ii)+'.png')
        plt.clf()
        plt.close()
    return

def pre_process(data,max_data=None):
    transformed_data = np.zeros_like(data)
    #First convert to Polar Space
    #Normalize
    if max_data is None:
        max_data = np.max(data,axis=1).max(axis=0)
    data = data/max_data
    #Rho
    #transformed_data[:,:,0] = np.sqrt(data[:,:,0]**2 + data[:,:,1]**2)
    #transformed_data[:,:,1] = np.arcsin(data[:,:,1]/transformed_data[:,:,0])
    #transformed_data[:,:,2] = data[:,:,2]

    #return transformed_data,max_data
    return data,max_data

        
