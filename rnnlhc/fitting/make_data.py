import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

MaxNumSteps = 12
def generate_data(num=None,data_type=1):
        x_data_list = np.zeros((num,MaxNumSteps+1))
        y_data_list = np.zeros((num,MaxNumSteps+1))
        for j in range(0,num):
                rand_freq = np.random.rand()
                #generate randome sequence equal to number of points
                x_data = np.array([np.linspace(0,4*np.pi,num = MaxNumSteps+1)] )
                if data_type == 1: #Sinusoid
                    y_data = sinusoid(x_data)
                elif data_type == 2: #Sinusoid with large noise
                    y_data = sinusoid2(x_data)
                else:
                    y_data = sinusoid3(x_data,rand_freq)
                x_data_list[j,:] = x_data
                y_data_list[j,:] = y_data
        
        return x_data_list,y_data_list

def sinusoid3(x,rand_freq):
    return 0.5*np.sin(x*rand_freq) + np.random.rand()/100.

def sinusoid2(x):
    return 0.5*np.sin(x*np.random.rand()/10.) + np.random.rand()/100.
    #return 0.5*np.sin(x) + 0.25

def sinusoid(x):
    return 0.5*np.sin(x) + np.random.rand()/10.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates Training and Test Data")
    parser.add_argument("--nTrain",type=int,default=1000,help="Number of Training Samples")
    parser.add_argument("--nTest",type=int,default=500,help="Number of Test Samples")
    parser.add_argument("--dType",type=int,default=3,help="The type of data generation mode")
    args = parser.parse_args()
    ind,train = generate_data(args.nTrain,data_type=args.dType)
    ind,test = generate_data(args.nTest,data_type=args.dType)
    pickle.dump([ind,train,test],open('../data/rand_fixed_freq_sinusoids.pkl','w'))
