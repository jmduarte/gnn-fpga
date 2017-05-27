import numpy as np
from utilities import calc_eta, calc_phi, filter_samples, filter_objects

class ActsData():
    """Empty class for setting variables as attributes"""
    pass

def load_data(filename):
    """
    Retrieve data from one file
    Returns a data object with attributes for each numpy array
    """
    d = ActsData()
    f = np.load(filename, encoding='bytes')
    # Track level truth quantities
    d.true_theta = f['truth_Theta']
    d.true_eta = calc_eta(d.true_theta)
    d.true_phi = f['truth_Phi']
    d.true_qop = f['truth_QoverP']
    d.true_pt = np.abs(1/d.true_qop)
    # Detector hit measurements
    d.nstep = f['Filter_nSteps']
    d.rphi = f['Meas_RPHI']
    d.z = f['Meas_z']
    d.r = f['Cyl_R']
    d.phi = calc_phi(d.rphi, d.r)
    d.KF_z = f['Filter_z']
    d.KF_phi = f['Filter_Phi']
    d.KF_r = f['Filter_R']
    return d

def clean_data(data, fix_phi=False):
    """
    Cleans up the data, selecting barrel tracks and good hits.
    """
    barrel_tracks = np.abs(data.true_eta) < 1
    d = ActsData()

    # filter out all tracks not perfectly in the barrel.
    d.true_theta, d.true_eta, d.true_phi, d.true_qop, d.true_pt = (
        filter_samples(barrel_tracks, data.true_theta, data.true_eta,
                       data.true_phi, data.true_qop, data.true_pt))
    d.nstep, d.rphi, d.z, d.r, d.phi = (
        filter_samples(barrel_tracks, data.nstep, data.rphi,
                       data.z, data.r, data.phi))
    d.KF_z, d.KF_phi, d.KF_r = (
        filter_samples(barrel_tracks, data.KF_z, data.KF_phi,
                       data.KF_r))

    # To select the actual layer hits, I select the indices of the steps
    # I want. I'm currently taking the middle of each detector layer triplet,
    # and ignoring all of the apparent "auxiliary" steps. This assumes
    # all tracks have the fixed 31 steps as previously discovered, so it's
    # a bit fragile and will need to be updated if the data changes.
    assert np.all(d.nstep == 31)
    #good_hit_idxs = np.array([1, 4, 9, 11, 14, 17, 20, 24, 27])
    good_hit_idxs = np.array([2, 5, 8, 11, 15, 18, 21, 25, 28])
    d.rphi, d.z, d.r, d.phi = filter_objects(
        good_hit_idxs, d.rphi, d.z, d.r, d.phi)
    d.KF_r, d.KF_z, d.KF_phi = filter_objects(
        good_hit_idxs, d.KF_r, d.KF_z,d.KF_phi)
    
    # Current data has some funny artifacts in phi.
    # Here is a shitty, hacky correction. Needs to be fixed upstream.
    if fix_phi:
        for i in range(d.phi.shape[1]):
            phi = d.phi[:,i]
            phi = phi * np.pi * 2 / (phi.max() - phi.min())
            d.phi[:,i] = phi - phi.min() - np.pi
        for i in range(d.KF_phi.shape[1]):
            KF_phi = d.KF_phi[:,i]
            KF_phi = KF_phi * np.pi * 2 / (KF_phi.max() - KF_phi.min())
            d.KF_phi[:,i] = KF_phi - KF_phi.min() - np.pi

    # Calculate theta
    d.theta = np.arctan(d.r / d.z)
    # Fix negative values so theta ranges from (0, pi)
    negidx = d.theta < 0
    d.theta[negidx] = d.theta[negidx] + np.pi
    d.eta = calc_eta(d.theta)

    return d
