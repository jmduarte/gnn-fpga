"""
Conv approach for the 3D detector model case.

"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.cm as cm

def gen_noise(shape, prob=0.1, seed_layers=0):
    """Generate uniform noise data of requested shape"""
    noise = (np.random.random_sample(shape) < prob).astype(np.int8)
    noise[:,:seed_layers,:,:] = 0
    return noise

def sample_track_params(n, num_det_layers, det_layer_size):
    """Generate track parameters constrained within detector shape"""
    # Sample the entry and exit points for tracks
    entry_points = np.random.uniform(0, det_layer_size, size=(n, 2))
    exit_points = np.random.uniform(0, det_layer_size, size=(n, 2))
    # Calculate slope parameters
    slopes = (exit_points - entry_points) / float(num_det_layers - 1)
    #print("\nslopes: ", slopes, "\n\n")
    #print("\nentry_points: ", entry_points, "\n\n")
    #print("\nexit_points: ", exit_points, "\n\n")
    return np.concatenate([slopes, entry_points], axis=1)

def track_hit_coords(params, det_layer_idx=None, num_det_layers=None, as_type=np.int):
    """
    Given an array of track params, give the coordinates
    of the hits in detector index space
    """
    if det_layer_idx is None:
        det_layer_idx = np.arange(num_det_layers)
    xslope, yslope, xentry, yentry = params
    xhits = xslope*det_layer_idx + xentry
    yhits = yslope*det_layer_idx + yentry
    return xhits.astype(as_type), yhits.astype(as_type), xhits, yhits

def gen_straight_tracks(n, num_det_layers, det_layer_size):
    """Generate n straight tracks"""
    # Initialize the data
    data = np.zeros((n, num_det_layers, det_layer_size, det_layer_size),
                    dtype=np.float32)
    # Sample track parameters
    params = sample_track_params(n, num_det_layers, det_layer_size)
    # Calculate hit positions and fill hit data
    idx = np.arange(num_det_layers)
    if (len(params)==0): a=[]
    else: a = np.zeros(shape=(n, 2*num_det_layers))
    for ievt in range(n):
        xhits, yhits, xfhits, yfhits = track_hit_coords(params[ievt], idx)
        #data[ievt,idx,xhits,yhits] = 1
        data[ievt,idx,(det_layer_size-1)-yhits,xhits] = 1
        hit_index_tmp = np.zeros(shape=(2*num_det_layers))
        count_tmp = 0
        for i in range(0, len(xhits)):
            hit_index_tmp[count_tmp] = xfhits[i]
            hit_index_tmp[count_tmp+1] = yfhits[i]
            count_tmp = count_tmp+2
        if (len(params)!=0):
            tmp = np.concatenate([hit_index_tmp], 0)
            a[ievt] = tmp
    return data, params, a

def gen_bkg_tracks(num_event, num_det_layers, det_layer_size,
                   avg_bkg_tracks=3, seed_layers=0):
    """
    Generate background tracks in the non-seed detector layers.
    Samples the number of tracks for each event from a poisson
    distribution with specified mean avg_bkg_tracks.
    """
    num_bkg_tracks = np.random.poisson(avg_bkg_tracks, num_event)
    bkg_tracks = np.zeros((num_event, num_det_layers, det_layer_size, det_layer_size),
                          dtype=np.float32)
    for ievt in range(num_event):
        ntrk = num_bkg_tracks[ievt]
        bkg_tracks[ievt] = sum(gen_straight_tracks(ntrk, num_det_layers, det_layer_size)[0])
    bkg_tracks[:,:seed_layers,:,:] = 0
    return bkg_tracks

def draw_from_params(params, num_det_layers, det_layer_size):
    params = params[0]
    layer_idx = np.arange(num_det_layers)
    x, y, tmp1, tmp2 = track_hit_coords(params, layer_idx, as_type=np.float32)
    xhits, yhits = np.rint(x), np.rint(y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(layer_idx, xhits+0.5, yhits+0.5)
    ax.scatter(layer_idx, x, y)
    ax.plot(layer_idx, x, y)
    ax.set_xlim(0, num_det_layers-1)
    ax.set_ylim(0, det_layer_size)
    ax.set_zlim(0, det_layer_size)
    ax.set_xlabel('detector layer')
    ax.set_ylabel('pixel x')
    ax.set_zlabel('pixel y')
    plt.tight_layout();

    
def drawMulti_from_params(num_tracks, params, num_det_layers, det_layer_size, target=False):
    layer_idx = np.arange(num_det_layers)
    xl, yl = [], []
    xlhits, ylhits = [], []
    for i in params:
        x, y = track_hit_coords(i, layer_idx, as_type=np.float32)
        xhits, yhits = x.astype(np.int), y.astype(np.int)
        xlhits.append(xhits)
        ylhits.append(yhits)
        xl.append(x)
        yl.append(y)

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, num_tracks):
        if(target): ax.scatter(layer_idx, xlhits[i]+0.5, ylhits[i]+0.5)
        ax.plot(layer_idx, xl[i], yl[i])
    ax.set_xlim(0, num_det_layers-1)
    ax.set_ylim(0, det_layer_size)
    ax.set_zlim(0, det_layer_size)
    ax.set_xlabel('detector layer')
    ax.set_ylabel('pixel x')
    ax.set_zlabel('pixel y')
    plt.tight_layout();


def drawMulti_from_params_pix(num_tracks, params, num_det_layers, det_layer_size, target=False):
    #print(params[:][:4])
    layer_idx = np.arange(num_det_layers)
    xl, yl = [], []
    xlhits, ylhits = [], []
    colors = cm.rainbow(np.linspace(0, 1, num_tracks))
    
    if(target):
        for i in range(0, num_tracks):
            x, y, tmp1, tmp2 = track_hit_coords(params[i][:4], layer_idx, as_type=np.float32)
            xhits, yhits = x.astype(np.int), y.astype(np.int)
            xlhits.append(xhits)
            ylhits.append(yhits)
            xl.append(x)
            yl.append(y)
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, c in zip(range(0, num_tracks), colors):
            ax.scatter(layer_idx, xlhits[i]+0.5, ylhits[i]+0.5, color=c)
            ax.plot(layer_idx, xl[i], yl[i], color=c)
        ax.set_xlim(0, num_det_layers-1)
        ax.set_ylim(0, det_layer_size)
        ax.set_zlim(0, det_layer_size)
        ax.set_xlabel('detector layer')
        ax.set_ylabel('pixel x')
        ax.set_zlabel('pixel y')
        plt.tight_layout();
    
    else:
        for i in range(0, num_tracks):
            xhits = params[i][0::2]
            yhits = params[i][1::2]
            xlhits.append(xhits)
            ylhits.append(yhits)
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, c in zip(range(0, num_tracks), colors):
            #ax.scatter(layer_idx, xlhits[i]+0.5, ylhits[i]+0.5, color=c)
            ax.scatter(layer_idx, xlhits[i], ylhits[i], color=c)
        ax.set_xlim(0, num_det_layers-1)
        ax.set_ylim(0, det_layer_size)
        ax.set_zlim(0, det_layer_size)
        ax.set_xlabel('detector layer')
        ax.set_ylabel('pixel x')
        ax.set_zlabel('pixel y')
        plt.tight_layout();

    
def generate_data(shape, num_seed_layers=3, avg_bkg_tracks=3,
                  noise_prob=0.01, verbose=True):
    """
    Top level function to generate a dataset.
    
    Returns arrays (events, sig_tracks, sig_params)
    """
    num_event, num_det_layers, det_layer_size, _ = shape
    # Signal tracks
    sig_tracks, sig_params, a = gen_straight_tracks(
        num_event, num_det_layers, det_layer_size)
    # Background tracks
    bkg_tracks = gen_bkg_tracks(
        num_event, num_det_layers, det_layer_size,
        avg_bkg_tracks=avg_bkg_tracks, seed_layers=num_seed_layers)
    # Noise
    noise = gen_noise(shape, prob=noise_prob, seed_layers=num_seed_layers)
    # Full events
    events = sig_tracks + bkg_tracks + noise
    events[events > 1] = 1
    # Print data sizes
    if verbose:
        print('Sizes of arrays')
        print('  events:     %g MB' % (events.dtype.itemsize * events.size / 1e6))
        print('  sig_tracks: %g MB' % (sig_tracks.dtype.itemsize * sig_tracks.size / 1e6))
        print('  bkg_tracks: %g MB' % (bkg_tracks.dtype.itemsize * bkg_tracks.size / 1e6))
        print('  noise:      %g MB' % (noise.dtype.itemsize * noise.size / 1e6))
        print('  sig_params: %g MB' % (sig_params.dtype.itemsize * sig_params.size / 1e6))
    return events, sig_tracks, sig_params, a


def get_Alist_pulls(model_2, epoch_size, train_events, train_weights, 
                    train_targets_slope_pix, target_index, plot):
    b1_pred, b1_target = [], []
    for i in range(0, epoch_size):
        test_event = train_events[i]
        test_weights = train_weights[i].astype(np.bool_)
        test_target = train_targets_slope_pix[i][test_weights]
        test_pred = model_2.predict(np.asarray([test_event]))[0][test_weights]
        for b1 in test_pred:
            b1_pred.append(b1[target_index])
        for b1 in test_target:
            b1_target.append(b1[target_index])
            
    b1_res = []
    for (b1_pred_tmp, b1_target_tmp) in zip(b1_pred, b1_target):
        b1_res.append(b1_target_tmp - b1_pred_tmp)
    
    (mu, sigma) = norm.fit(b1_res)
    if(plot):
        n, bins, patches = plt.hist(b1_res, bins='sqrt', normed=1, facecolor='green', alpha=0.75)
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=1)
        plt.xlabel('Slope (truth-predict)')
        plt.ylabel('')
        plt.title(r'$\mathrm{Residual\ distribution}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
        plt.grid(True)
        plt.show()
    return (mu, sigma)
