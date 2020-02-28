import uproot
import awkward
import numpy as np
import argparse
import os,glob
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
#from tqdm import tqdm_notebook as tqdm

from Muon_graph  import Graph,SparseGraph, make_sparse_graph, \
				  save_graph, save_graphs, load_graph, \
				  load_graphs, make_sparse_graph, graph_from_sparse, \
				  draw_sample_withproperties

# Decide EMTF hit layer number
emtf_lut = np.zeros((5,5,5), dtype=np.int32) - 99

   #Type #Station #Ring   # Layer
##emtf_lut[1,1,4]  = 0  # ME1/1a
##emtf_lut[1,1,1]  = 0  # ME1/1b
##emtf_lut[1,1,2]  = 1  # ME1/2
##emtf_lut[1,1,3]  = 1  # ME1/3
##emtf_lut[1,2,1]  = 2  # ME2/1
##emtf_lut[1,2,2]  = 2  # ME2/2
##emtf_lut[1,3,1]  = 3  # ME3/1
##emtf_lut[1,3,2]  = 3  # ME3/2
##emtf_lut[1,4,1]  = 4  # ME4/1
##emtf_lut[1,4,2]  = 4  # ME4/2
##emtf_lut[2,1,2]  = 5  # RE1/2
##emtf_lut[2,2,2]  = 6  # RE2/2
##emtf_lut[2,3,1]  = 7  # RE3/1
##emtf_lut[2,3,2]  = 7  # RE3/2
##emtf_lut[2,3,3]  = 7  # RE3/3
##emtf_lut[2,4,1]  = 8  # RE4/1
##emtf_lut[2,4,2]  = 8  # RE4/2
##emtf_lut[2,4,3]  = 8  # RE4/3
##emtf_lut[3,1,1]  = 9  # GE1/1
##emtf_lut[3,2,1]  = 10 # GE2/1
##emtf_lut[4,1,1]  = 11 # ME0

emtf_lut[1,1,4] = 2  # ME1/1a
emtf_lut[1,1,1] = 2  # ME1/1b
emtf_lut[1,1,2] = 3  # ME1/2
emtf_lut[1,1,3] = 3  # ME1/3
emtf_lut[1,2,1] = 7  # ME2/1
emtf_lut[1,2,2] = 7  # ME2/2
emtf_lut[1,3,1] = 8  # ME3/1
emtf_lut[1,3,2] = 8  # ME3/2
emtf_lut[1,4,1] = 10  # ME4/1
emtf_lut[1,4,2] = 10  # ME4/2
emtf_lut[2,1,2] = 4  # RE1/2
emtf_lut[2,2,2] = 5  # RE2/2
emtf_lut[2,3,1] = 9  # RE3/1
emtf_lut[2,3,2] = 9  # RE3/2
emtf_lut[2,3,3] = 9  # RE3/3
emtf_lut[2,4,1] = 11  # RE4/1
emtf_lut[2,4,2] = 11  # RE4/2
emtf_lut[2,4,3] = 11  # RE4/3
emtf_lut[3,1,1] = 1  # GE1/1
emtf_lut[3,2,1] = 6 # GE2/1
emtf_lut[4,1,1] = 0 # ME0


def get_layer(type, station, ring):
    return emtf_lut[int(type),int(station),int(ring)]

def get_layers(types,stations,rings):
	array = []
	for i in range(0, len(types)):
		array.append([get_layer(types[i],stations[i],rings[i])]  )
	return array

def column(matrix, i):
    return [int(row[i]) for row in matrix]

def plotgraph(file,outputname,output):
	g1sparse = load_graph('%s'%file,graph_type=SparseGraph)
	g1    = graph_from_sparse(g1sparse)
	layer = get_layers(column(g1.X,9),column(g1.X,7),column(g1.X,8))
	Xnew  = np.hstack((g1.X,layer))
	findindex = find(np.rot90(g1.Ri))
	x,y,z = findindex
	if g1.pt>20 and abs(g1.eta)<2.4 and abs(g1.eta)>1.2:
		draw_sample_withproperties(Xnew,g1.Ri,g1.Ro,g1.y,g1.pt,g1.eta,skip_false_edges=False,outputname=outputname,output=output)

def plotgraphs(directory,outputname,ngraphs):
	os.system("mkdir myplots")
	os.system("mkdir myplots/%s"%outputname)	
	files = glob.glob("%s/*.npz"%directory)
	i=0
	for file in files: 
		if i<int(ngraphs): plotgraph(file,outputname,i)
		i+=1
#############COMMAND CODE IS BELOW ######################

###########OPTIONS
parser = argparse.ArgumentParser(description='Command line parser of skim options')
parser.add_argument('--directory',    dest='directory',  help='Name of directory with the graphs', required = True)
parser.add_argument('--outputname',    dest='outputname',  help='Name of directory with the graphplots', required = True)
parser.add_argument('--ngraphs',    dest='ngraphs',  help='Number of graphs to plot', required = True)
parser.set_defaults(lepveto=False)
args = parser.parse_args()
directory       = args.directory
outputname      = args.outputname
ngraphs         = args.ngraphs
#Plot all graphs
plotgraphs(directory,outputname,ngraphs)