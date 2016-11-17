"""
This module contains helper code for generating and manipulating toy detector
data for the ML algorithms.

So far it just has the 1D detector straight-track data generation.
"""

import numpy as np

def simulate_straight_track(m, b, det_shape):
    """
    Simulate detector data for one straight track.
    Parameters:
        m: track slope parameter
        b: track first-layer intercept parameter
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of binary detector data for one track.
    """
    x = np.zeros(det_shape)
    hit_idxs = [round(m*l + b) for l in range(det_shape[0])]
    for row, idx in enumerate(hit_idxs):
        x[row, idx] = 1
    return x

def generate_straight_track(det_shape):
    """
    Sample track parameters and simulate detector data.
    Parameters:
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of binary detector data for one track.
    """
    det_depth, det_width = det_shape
    # Sample detector entry point
    b = np.random.random_sample()*(det_width - 1)
    # Sample detector exit point
    b2 = np.random.random_sample()*(det_width - 1)
    # Calculate track slope
    m = (b2 - b) / det_depth
    # restrict slope to only generate tracks that traverse the entire detector
    #mmax = (det_width - 1 - b) / (det_depth - 1)
    #mmin = -b / det_depth
    #m = np.random.random_sample() * (mmax - mmin) + mmin
    return simulate_straight_track(m, b, det_shape)

def generate_straight_tracks(n, det_shape):
    """
    Generates single straight-track events.
    Parameters:
        n: number of single-track events to generate
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of detector data for n single-track events. The shape is
        (n, det_shape[0], det_shape[1]).
    """
    tracks = [np.expand_dims(generate_straight_track(det_shape), 0)
              for i in range(n)]
    return np.concatenate(tracks, axis=0)

def generate_uniform_noise(n, det_shape, prob=0.1, skip_layers=5):
    """
    Generate uniform noise hit data.
    Parameters:
        n: number of noise events to generate
        det_shape: tuple of detector shape: (depth, width)
        prob: probability of noise hit in each pixel
        skip_layers: number of detector layers to skip (no noise)
    Returns:
        ndarray of detector noise data for n events. The shape is
        (n, det_shape[0], det_shape[1]).
    """
    # One way to do this: generate random floats in [0,1]
    # and then convert the ones above threshold to binary
    det_depth, det_width = det_shape
    noise_events = np.zeros([n, det_depth, det_width])
    rand_vals = np.random.random_sample([n, det_depth-skip_layers, det_width])
    noise_events[:,skip_layers:,:] = rand_vals < prob
    return noise_events

def generate_track_bkg(n, det_shape, tracks_per_event=2, skip_layers=5):
    """
    Generate events with a number of clean layers followed by layers containing
    background tracks.

    Parameters:
        n: number of events to generate
        det_shape: tuple of detector shape: (depth, width)
        tracks_per_event: fixed number of tracks to simulate in each event
        skip_layers: number of detector layers to skip (no bkg)
    Returns:
        ndarray of detector data for n events. The shape is
        (n, det_shape[0], det_shape[1]).
    """
    # Combine single-track events to make multi-track events
    events = sum(generate_straight_tracks(n, det_shape)
                 for i in range(tracks_per_event))
    # Zero out the skipped layers
    events[:,0:skip_layers,:] = 0
    return events
