"""
This module contains helper code for generating and manipulating toy detector
data for the ML algorithms.

So far it just has the 1D detector straight-track data generation.
"""

import numpy as np
import matplotlib.pyplot as plt

def draw_event(event, title=None, mask_ranges=None, **kwargs):
    """
    Draw and format one 1D detector event with matplotlib.
    Params:
        event: data for one event in image format
        title: plot title
        mask_range: tuple of arrays, (lower, upper) defining a detector
            mask envelope that will be drawn on the display
        kwargs: additional keywords passed to pyplot.plot
    """
    plt.imshow(event.T, interpolation='none', aspect='auto',
               origin='lower', **kwargs)
    if title is not None:
        plt.title(title)
    plt.xlabel('Detector layer')
    plt.ylabel('Detector pixel')
    plt.autoscale(False)
    plt.tight_layout()
    if mask_ranges is not None:
        plt.plot(mask_ranges[0], 'w:')
        plt.plot(mask_ranges[1], 'w:')

def calc_mask_ranges(det_width, mask_shapes):
    """
    Calculate the indices of the detector mask envelope.
    Parameters:
        det_width: width of the 1D detector
        mask_shapes: ndarray of widths of the mask
    Returns:
        Two arrays representing the lower and upper index ranges of the detector mask.
    """
    lower = ((det_width - mask_shapes) / 2).astype(int)
    upper = lower + mask_shapes
    return lower, upper

def construct_mask(det_shape, mask_shapes):
    """
    Construct the boolean mask used to select a wedge of the detector.
    Parameters:
        det_shape: shape of the full 1D detector
        mask_shapes: ndarray of widths of the mask
    Returns:
        Boolean array of the detector mask, with dimensions matching det_shape.
    """
    det_mask = np.zeros(det_shape, bool)
    lower, upper = calc_mask_ranges(det_shape[1], mask_shapes)
    for i, (low, up) in enumerate(zip(lower, upper)):
        det_mask[i, low:up] = True
    return det_mask

def apply_det_mask(data, mask):
    """
    Apply detector mask to 1D detector data events.
    Parameters:
        data: ndarray of 1D detector events
        mask: boolean detector mask ndarray
    Returns:
        List of masked layer data arrays.
    """
    assert data[0].shape == mask.shape, \
        'shapes unequal: {} != {}'.format(data[0].shape, mask.shape)
    # Group event data by masked layers
    return [data[:,ilayer,mask[ilayer]] for ilayer in range(mask.shape[0])]

def expand_masked_data(masked_data, mask):
    """
    Unmask detector data and expand into fixed-size detector array.
    Parameters:
        masked_data: list of ndarrays of detector layer data
            for multiple events
        mask: boolean detector mask used to mask the data
    Returns:
        ndarray of data where each event is same shape as the mask
    """
    # Let's first assume that all layers are present.
    # I will still need to handle the case where first or last layer is dropped.
    assert len(masked_data) == mask.shape[0], \
        'Data shape incompatible with detector mask'
    output_shape = (len(masked_data[0]), *mask.shape)
    output = np.zeros(output_shape)
    # Loop over layers
    for ilayer, mask in enumerate(mask):
        output[:,ilayer,mask] = masked_data[ilayer]
    return output

def simulate_straight_track(m, b, det_shape):
    """
    Simulate detector data for one straight track.
    Parameters:
        m: track slope parameter
        b: track first-layer intercept parameter (detector entry point)
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of binary detector data for one track.
    """
    x = np.zeros(det_shape)
    idx = np.arange(det_shape[0])
    hits = (idx*m + b).astype(int)
    x[idx, hits] = 1
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
