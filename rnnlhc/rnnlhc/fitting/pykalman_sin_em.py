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

from pykalman import KalmanFilter

rnd = np.random.RandomState(0)

# generate a noisy sine wave to act as our fake observations
n_timesteps = 100
x = np.linspace(0, 3 * np.pi, n_timesteps)
observations = 20 * (np.sin(x) + 0.05 * rnd.randn(n_timesteps))
n_states = 10

# create a Kalman Filter by hinting at the size of the state and observation
# space.  If you already have good guesses for the initial parameters, put them
# in here.  The Kalman Filter will try to learn the values of all variables.
#kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
#                  transition_covariance=0.01 * np.eye(2),
#                  em_vars=['transition_matrices','transition_covariance',
#                  'observation_matrices', 'observation_covariance',
#                  'observation_offsets','transition_offsets'])
kf = KalmanFilter(transition_matrices=np.random.randn(n_states,n_states),
                  transition_covariance=0.01 * np.eye(n_states),
                  em_vars=['transition_matrices','transition_covariance',
                  'observation_matrices', 'observation_covariance',
                  'observation_offsets','transition_offsets'])

# You can use the Kalman Filter immediately without fitting, but its estimates
# may not be as good as if you fit first.
import IPython; IPython.embed()
state_means, state_covariances = kf.em(observations,n_iter=10).filter(observations)
obs_pred = np.zeros_like(observations)
for ii in np.arange(observations.shape[0]):
    obs_pred[ii] = kf.observation_matrices.dot(state_means[ii])+ kf.observation_offsets + np.random.multivariate_normal(np.zeros(1),kf.observation_covariance)
print('fitted model: {0}'.format(kf))

# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.
pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, obs_pred,
                        linestyle='-', marker='o', color='r',
                        label='KF_predictions')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=x.max())
pl.xlabel('time')
pl.savefig('pykalman_sin.png')
