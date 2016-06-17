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

