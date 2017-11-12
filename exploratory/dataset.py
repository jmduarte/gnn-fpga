"""TrackML dataset loading"""

__authors__ = ['Moritz Kiehn', 'Sabrina Amrouche']

import glob
import os.path
import re

import numpy

DTYPE_HITS = numpy.dtype([
    ('hit_id', 'i8'),
    ('volume_id', 'i4'),
    ('layer_id', 'i4'),
    ('module_id', 'i4'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('z','f4'),
    ('ex', 'f4'),
    ('ey', 'f4'),
    ('ez','f4'),
    ('phi', 'f4'),
    ('theta', 'f4'),
    ('ephi', 'f4'),
    ('etheta', 'f4'),
    ('ncells', 'i4'), ])
DTYPE_PARTICLES = numpy.dtype([
    ('particle_id', 'i8'),
    ('vx', 'f4'),
    ('vy', 'f4'),
    ('vz', 'f4'),
    ('px', 'f4'),
    ('py', 'f4'),
    ('pz', 'f4'),
    ('q', 'i4') ])
DTYPE_MAPPING = numpy.dtype([('hit_id', 'i8'), ('particle_id', 'i8')])

def load_event(prefix):
    """
    Load the full data for a single event with the given prefix.
    Returns a tuple (hits, particles, truth) where particles and truth
    can be None. Each element is a numpy structured array with field names
    identical to the CSV column names and appropriate types.
    All output arrays are sorted first by hit_id and then by particle_id.
    """
    print(prefix)

    file_hits= glob.glob(prefix+"-hits.csv*")
    if len(file_hits)==0:
        raise Exception("No file found matching hits.csv* with prefix:"+prefix)
    elif len(file_hits)>1:
        raise Exception("More than one file found matching hits.csv* with prefix:"+prefix)
    else:
        file_hits=file_hits[0]
    hits = numpy.loadtxt(file_hits, dtype=DTYPE_HITS,
                     delimiter=',', skiprows=1, usecols=list(range(15)))
    hits.sort(order='hit_id')

    file_particles= glob.glob(prefix+"-particles.csv*")
    if len(file_particles)==0:
        particles = None  # misssing particles file is not fatal
        print("Warning : no file found matching particles.csv* with prefix:"+prefix)
    elif len(file_particles)>1:
        raise Exception("More than one file found matching particles.csv* with prefix:"+prefix)
    else:
        file_particles=file_particles[0]
        particles = numpy.loadtxt(file_particles, dtype=DTYPE_PARTICLES,
                          delimiter=',', skiprows=1)
        particles.sort(order='particle_id')

    file_truth = glob.glob(prefix+"-truth.csv*")
    if len(file_truth)==0:
        truth = None # misssing truth file is not fatal
        print("Warning : no file found matching truth.csv* with prefix:"+prefix)
    elif len(file_truth)>1:
        raise Exception("More than one file found matching truth.csv* with prefix:"+prefix)
    else:
        file_truth = file_truth[0]
        truth = numpy.loadtxt(file_truth, dtype=DTYPE_MAPPING,
                          delimiter=',', skiprows=1)
        truth.sort(order=['hit_id', 'particle_id'])

    return hits, particles, truth

def load_dataset(path):
    """
    Provide an iterator over all events in a datset directory.
    For each event it returns a tuple (name, hits, particles, truth) where
    particles and truth can be None.
    """
    # each event must have a hits files
    hits_files = glob.glob(os.path.join(path, 'event*-hits.csv*'))
    for f in hits_files:
        name = os.path.basename(f).split('-', maxsplit=1)[0]
        data = (name,) + load_event(os.path.join(path, name))
        yield data