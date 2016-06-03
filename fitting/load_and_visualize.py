import json
import StringIO
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname",type=str,default="../data/EventDump_10tracks.json",help="Name of the json file")
    parsed_args = parser.parse_args()
    #First Read the file
    json_data = open(parsed_args.fname,'r').read()
    parsed_json_data = json.loads(json_data)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for ii in np.arange(10):
      data = parsed_json_data['xAOD::Type::TrackParticle']['InDetTrackParticles']['Trk '+str(ii)]['pos']
      data = np.array(data)
      ax.plot(data[:,0],data[:,1],data[:,2], linestyle='--',linewidth=3.2,label='Track 0')
      ax.view_init(elev=18, azim=-27)
      ax.hold(True)
    ax.dist=9
    plt.show()
