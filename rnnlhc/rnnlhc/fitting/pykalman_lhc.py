'''
==================================
Kalman Filter tracking a sine wave
==================================

This example shows how to use the Kalman Filter for state estimation.

In this example, we generate a fake target trajectory using a sine wave.
Instead of observing those positions exactly, we observe the position plus some
random noise.  We then use a Kalman Filter to estimate the velocity of the
system as well.

The figure drawn illustrates the observations, and the position and velocity
estimates predicted by the Kalman Smoother.
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
from rnnlhc.fitting.BatchData import BatchNpyData
from utilities import proj_2d_plot,pre_process
import argparse
from pykalman import KalmanFilter

def init():
    return 1e-1*np.random.rand()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for fitting")
    parser.add_argument("--niter",default=10,type=int,help="Number of iterations")
    parser.add_argument("--json_data",default='../data/EventDump_10Ktracks.json',type=str,help="Json data path")
    parser.add_argument("--npy_data",default='../data/ET_muons_10K_0000.npy',type=str,help="NPY data")
    args = parser.parse_args()
    rnd = np.random.RandomState(0)
    data = np.load(args.npy_data)
    BD= BatchNpyData(data)
    n_states = 5
    n_dim_obs = 3
    MaxNumSteps = 10
    batch_size = 200
    test, rand_int = BD.sample_batch(MaxNumSteps,batch_size)
    test, _ = pre_process(test)

    # create a Kalman Filter by hinting at the size of the state and observation
    # space.  If you already have good guesses for the initial parameters, put them
    # in here.  The Kalman Filter will try to learn the values of all variables.
    #kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
    #                  transition_covariance=0.01 * np.eye(2),
    #                  em_vars=['transition_matrices','transition_covariance',
    #                  'observation_matrices', 'observation_covariance',
    #                  'observation_offsets','transition_offsets'])
    trans_mat_init = np.array([[1,0,init(),init(),init()],
    [0,1,init(),init(),init()],
    [init(),init(),1,0,0],
    [init(),init(),0,1,0],
    [init(),init(),0,0,1]])
    observation_mat = np.array([[1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0]])
    kf = KalmanFilter(#transition_matrices=1e-3*np.eye(n_states,n_states),
                      transition_matrices = trans_mat_init,
                      transition_covariance=1e-1 * np.eye(n_states),
                      observation_matrices = observation_mat,
                      #transition_offsets = 1e-3*np.eye(MaxNumSteps,n_states),
                      #observation_offsets = 1e-2*np.eye(MaxNumSteps,3),
                      observation_covariance=1e-1*np.eye(n_dim_obs),
                      n_dim_obs=n_dim_obs,n_dim_state=n_states,
                      em_vars=['transition_matrices',
                      'observation_matrices',
                      'observation_offsets','transition_offsets'])

    # You can use the Kalman Filter immediately without fitting, but its estimates
    # may not be as good as if you fit first.
    print("Fitting using EM sample by sample")
    for ii in np.arange(args.niter):
        print("ii is {}".format(ii))
        data,rand_int = BD.sample_batch(MaxNumSteps,100)
        data, max_data = pre_process(data)
        for jj in np.arange(100):
            print("jj is {}".format(jj))
            kf.em(data[jj,...],n_iter=1)
    print("Now proceeding to filtering")
    state_means = np.zeros((batch_size,MaxNumSteps,n_states))
    state_covariances = np.zeros((batch_size,MaxNumSteps,n_states,n_states))
    for ii in np.arange(batch_size):
        state_means[ii,...], state_covariances[ii,...] = kf.filter(test[ii,...])
    obs_pred = np.zeros((MaxNumSteps,batch_size,n_dim_obs))
    for ii in np.arange(test.shape[0]):
        obs_pred[:,ii,:] = (kf.observation_matrices.dot(state_means[ii].T) + \
        kf.observation_offsets.reshape(-1,1) + \
        np.random.multivariate_normal(np.zeros(n_dim_obs),kf.observation_covariance).reshape(-1,1)).T
    print('fitted model: {0}'.format(kf))

    # Plot lines for the observations without noise, the estimated position of the
    # target before fitting, and the estimated position after fitting.
    proj_2d_plot(test,obs_pred)
    import IPython; IPython.embed()
