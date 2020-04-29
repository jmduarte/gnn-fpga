import numpy as np
import argparse
import os,glob
from scipy.sparse import csr_matrix, find

from Muon_graph  import SparseGraphProp, load_graph,  \
				  load_graph, graph_from_sparse_prop, \
				  draw_sample_withproperties

from prepareMuonGraphs import emtf_lut

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
	g1sparse  = load_graph('%s'%file,graph_type=SparseGraphProp)
	g1        = graph_from_sparse_prop(g1sparse)
	layer     = get_layers(column(g1.X,9),column(g1.X,7),column(g1.X,8))
	Xnew      = np.hstack((g1.X,layer))
	findindex = find(np.rot90(g1.Ri))
	x,y,z = findindex
	#Here you can cut on the pt and eta of the generated muons and plot
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
parser.add_argument('--directory',  dest='directory',  help='Name of directory with the graphs',    required = True)
parser.add_argument('--outputname', dest='outputname', help='Name of directory for the graphplots', required = True)
parser.add_argument('--ngraphs',    dest='ngraphs',    help='Maximum number of graphs to plot',     required = True)
args = parser.parse_args()
directory       = args.directory
outputname      = args.outputname
ngraphs         = args.ngraphs
#Plot all graphs
plotgraphs(directory,outputname,ngraphs)