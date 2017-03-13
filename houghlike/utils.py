"""Utilities for training convolutional tracking models"""

# Standard imports
import sys

# Package imports
import numpy as np
import numpy.linalg
import scipy.stats as stats
import theano as th
import theano.tensor as T
from keras import models, layers, optimizers, backend as K
import matplotlib.pyplot as plt
import pylab

# Local imports
import gauss_likelihood_loss
from drawing import draw_event, draw_input_and_pred

# Default parameters
default_batch_size = 256
default_epoch_size = 1000*default_batch_size
default_det_shape = (50,50)

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
    valid = (hits > 0) & (hits < det_shape[1])
    x[idx[valid], hits[valid]] = 1
    return x

def gen_tracks(batch_size=default_batch_size, det_shape=default_det_shape):
    """Generator for single-track events.
       Arguments: 
         batch_size: number of events to yield for each call
       Yields: batches of training data for use with the keras fit_generator function
    """
    det_depth, det_width = det_shape
    slope_scale = det_width/2. # puts slopes and intercepts on a common scale 
    while True:
        # Entry and exit points are randomized
        bs = np.random.random_sample(size=batch_size)*det_width
        b2s = np.random.random_sample(size=batch_size)*det_width
        ms = (b2s-bs)/det_width*slope_scale # scaled slope
        tracks = np.zeros((batch_size, 1, det_depth, det_width))
        # append dummy values for covariance matrix entries
        # (these are predicted by the model but have no ground truth values)
        dummy_cov = np.zeros((batch_size))
        targets = zip(bs, ms, dummy_cov, dummy_cov, dummy_cov)
        for i, (b,m,_,_,_) in enumerate(targets):
            tracks[i,0] = simulate_straight_track(m/slope_scale, b, det_shape)
        targets = np.asarray(targets)
        yield tracks, targets
        
def gen_n_tracks(batch_size=default_batch_size, det_shape=default_det_shape, 
        n_tracks=1):
    """Generator for multi-track events.  
       Each event contains exactly n_tracks tracks.
       The target track parameters are sorted in 
        order of increasing intercept."""
    det_depth, det_width = det_shape
    gen_single = gen_tracks(batch_size=n_tracks, det_shape=det_shape)
    while True:
        batch_events = np.zeros((batch_size, 1, det_depth, det_width))
        batch_targets = -np.ones((batch_size, n_tracks, 5))
        for n in range(batch_size):
            tracks,targets = gen_single.next()
            batch_events[n,0] = np.clip( sum( tracks ), 0, 1)
            event_targets = np.asarray(targets)
            batch_targets[n] = event_targets[event_targets[:,0].argsort()] # sort by first column
        yield batch_events, batch_targets
        
# Generator for training track prediction model
def gen_n_tracks_nocov(batch_size=default_batch_size, det_shape=default_det_shape, 
        n_tracks=1):
    det_depth, det_width = det_shape
    gen_single = gen_tracks(batch_size=n_tracks, det_shape=det_shape)
    """Generates n-track events.  Output contains slopes and intercepts only;
        there are no dummy entries representing covariance matrix parameters."""
    while True:
        batch_events = np.zeros((batch_size, 1, det_depth, det_width))
        batch_targets = -np.ones((batch_size, n_tracks, 2))
        for n in range(batch_size):
            tracks,targets = gen_single.next()
            batch_events[n,0] = np.clip( sum( tracks ), 0, 1)
            event_targets = np.asarray(targets)[:,:2]
            batch_targets[n] = event_targets[event_targets[:,0].argsort()] # sort by first column
        yield batch_events, batch_targets

# I'm computing the covariance matrix using a theano function 
# in order to reuse the existing code.  It would be better to reimplement
# this using numpy arrays only.
in_params = T.vector()
out_params = gauss_likelihood_loss.covariance_matrix_2D( in_params )
covariance = th.function(inputs=[in_params], outputs=out_params)

def simulate_track_from_cov_matrix(track_params, cov_params, ntoys=1000,
        det_shape=default_det_shape):
    """Given the covariance matrix parameters, generate random track slopes/intercepts
        from the covariance matrix and draw the resulting tracks"""
    slope_scale = det_shape[1]/2.
    cov = covariance(cov_params)
    event = np.zeros(det_shape)
    toys = np.random.multivariate_normal(mean=track_params, cov=cov, size=ntoys)
    for i in range(ntoys):
        b, m = toys[i]
        event += simulate_straight_track(m/slope_scale, b, det_shape)
    return event

def simulate_event_from_cov_matrix(event_pred, ntoys=1000, det_shape=default_det_shape):
    return sum( [ simulate_track_from_cov_matrix(track_params=pred[:2], 
                        det_shape=det_shape, cov_params=pred[2:])
                        for pred in event_pred ] )

def make_pred_with_errors(model, n_tracks=1, det_shape=default_det_shape):
    """Shows the model's uncertainty graphically for a random event"""
    test_data = gen_n_tracks(n_tracks=n_tracks, det_shape=det_shape).next()
    test_event = test_data[0][0]
    test_target = test_data[1][0]
    test_pred = model.predict(np.asarray([test_event]))[0]
    print test_target
    print test_pred
    pred_event = simulate_event_from_cov_matrix(test_pred, ntoys=10000,
            det_shape=det_shape)
    draw_input_and_pred(test_event[0], pred_event)
    
def make_pred_without_errors(model, n_tracks=1, det_shape=default_det_shape):
    """Shows the tracks predicted by the model"""
    slope_scale = det_shape[1]/2.
    test_data = gen_n_tracks(n_tracks=n_tracks, det_shape=det_shape).next()
    test_event = test_data[0][0]
    test_target = test_data[1][0]
    test_pred = model.predict(np.asarray([test_event]))[0]
    print test_target
    print test_pred
    pred_event = np.clip(sum([ simulate_straight_track(
                    line[1]/slope_scale, line[0], det_shape) for line in test_pred ]), 0, 1)
    draw_input_and_pred(test_event[0], pred_event)

def build_conv_model(det_shape=default_det_shape, n_tracks=1):
    """Build current iteration of convolutional tracking model.
        Returns tuple:
          (full_model, track_pred_model, conv_model, pretrain_layers)
          where:
          -full_model is the entire model
          -track_pred_model is the part that predicts track parameters
            (excluding covariances)
          -conv_model is the convolutional part only
          -pretrain_layers is a list of layers for which trainable=False
            should be set after training the track-finding portion 
            of the model (if training that part separately)"""
    pretrain_layers = []

    input_layer = layers.Input(shape=(1, det_shape[0], det_shape[1]))
    layer = layers.Convolution2D(8, 3, 3, border_mode='same')(input_layer)
    pretrain_layers.append(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Convolution2D(8, 3, 3, border_mode='same')(layer)
    pretrain_layers.append(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.MaxPooling2D(pool_size=(2,2))(layer)
    layer = layers.Convolution2D(32, 3, 3, border_mode='same')(layer)
    pretrain_layers.append(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Convolution2D(32, 3, 3, border_mode='same')(layer)
    pretrain_layers.append(layer)
    layer = layers.Activation('relu')(layer)
    conv_model = models.Model(input=input_layer, output=layer)
    layer = layers.Flatten()(layer)

    layer_tracks = layers.Dense(400)(layer)
    pretrain_layers.append(layer_tracks)
    layer_tracks = layers.RepeatVector(n_tracks)(layer_tracks)
    layer_tracks = layers.LSTM(400, return_sequences=True)(layer_tracks)
    pretrain_layers.append(layer_tracks)
    output_layer_tracks = layers.TimeDistributed(layers.Dense(2))(layer_tracks) # track parameters
    pretrain_layers.append(output_layer_tracks)
    track_pred_model = models.Model(input=input_layer, output=output_layer_tracks)

    layer_cov = layers.Dense(400)(layer)
    layer_cov = layers.RepeatVector(n_tracks)(layer_cov)
    layer_cov = layers.LSTM(400, return_sequences=True)(layer_cov)
    layer_cov = layers.TimeDistributed(layers.Dense(3))(layer_cov) # track covariance matrix parameters
    output_layer_cov = layers.Lambda(gauss_likelihood_loss.covariance_from_network_outputs)(layer_cov)

    output_layer = layers.merge([output_layer_tracks, output_layer_cov], mode='concat', concat_axis=2)
    full_model = models.Model(input=input_layer, output=output_layer)

    return full_model, track_pred_model, conv_model, pretrain_layers
    

class PretrainableModel(object):
    """Class representing our tracking convnet model outputting both 
       track parameters and covariance matrices. Capable of pretraining
       the track-finding part and freezing that part while training the covariance part.
       Attributes:
           n_tracks (int): number of tracks per event
           epoch_size (int): number of events to process per training epoch
           pretrain (bool): whether to pretrain track-finding part
           pretrain_layers (list of keras layers): layers to pretrain
           track_pred_model (keras model): model predicting track parameters
           full_model (keras model): full model, including covariance matrix predictor
           conv_model (keras model): convolutional layers of the model
           track_pred_gen, full_gen: generators yielding batches of training data
               for training track_pred_model and full_model, respectively
           det_shape: tuple of integers (det_depth, det_width)
           model_fn: function used to construct the model. Signature should be:
                model_fn(det_shape, n_layers)
           """
    
    def __init__(self, n_tracks, pretrain=True, batch_size=default_batch_size, 
            epoch_size=default_epoch_size, det_shape=default_det_shape,
            model_fn=build_conv_model):
        self.n_tracks = n_tracks
        self.epoch_size = epoch_size
        self.pretrain = pretrain
        self.det_shape = det_shape
        self.track_pred_gen = gen_n_tracks_nocov(n_tracks=n_tracks, 
                det_shape=det_shape, batch_size=batch_size)
        self.full_gen = gen_n_tracks(n_tracks=n_tracks, 
                det_shape=det_shape, batch_size=batch_size)
        self.model_fn = model_fn
        self.set_model()
        
    def set_model(self):
        model_parts = self.model_fn(self.det_shape, self.n_tracks)
        # We try to unpack full/track/conv models and pretrain layers.
        # If we fail, assume the function returned track pred model only
        try:
            (self.full_model, self.track_pred_model, 
                    self.conv_model, self.pretrain_layers) = model_parts
        except TypeError: 
            self.track_pred_model = model_parts
            self.full_model, self.conv_model, self.pretrain_layers = None, None, None
        self.compile_track_pred_model()
        self.compile_full_model()
    
    def compile_track_pred_model(self):
        if self.track_pred_model:
            adam = optimizers.Adam(clipnorm=1.)
            self.track_pred_model.compile(loss='mean_squared_error', optimizer=adam)
        
    def compile_full_model(self):
        if self.full_model:
            adam = optimizers.Adam(clipnorm=1.)
            self.full_model.compile(loss=gauss_likelihood_loss.gauss_likelihood_loss_2D, 
                    optimizer=adam)
        
    def freeze_pretrain_layers(self, freeze=True):
        """If freeze=True, freeze weights on all pretrain layers.
           If freeze=False, unfreeze the weights.
           Important: the models need to be recompiled before this change will have any effect!"""
        for l in self.pretrain_layers:
            l.trainable = (not freeze)

    def train_track_pred_model(self, epochs=1):
        if self.pretrain: # if we don't pretrain the model, unfreezing layers is not necessary
            self.freeze_pretrain_layers(False)
        self.compile_track_pred_model()
        self.track_pred_model.fit_generator(self.track_pred_gen, self.epoch_size, epochs)
        
    def train_full_model(self, epochs=1):
        if self.pretrain:
            self.freeze_pretrain_layers()
        self.compile_full_model()
        self.full_model.fit_generator(self.full_gen, self.epoch_size, epochs)
        
    def make_pred_with_errors(self):
        make_pred_with_errors(self.full_model, n_tracks=self.n_tracks,
                det_shape=self.det_shape)
    
    def make_pred(self):
        make_pred_without_errors(self.full_model, n_tracks=self.n_tracks,
                det_shape=self.det_shape)

def squared_mahalanobis_distance(true, pred, cov):
    resid = pred-true
    # expand_dims is used to turn the row vector into a column vector
    precis_times_resid = np.matmul( np.linalg.inv(cov), np.expand_dims(resid, axis=-1) )
    return np.matmul( np.expand_dims(resid, axis=1), precis_times_resid )

def cov_from_params(params):
    """turns vector [variance1, covariance, variance2] into 2x2 covariance matrix"""
    return np.stack( [params, np.roll(params, shift=-1, axis=1)], axis=1 )[:,:,:2]

def sim_mahalanobis_distribution(model, ntoys=100, n_tracks=1,
        det_shape=default_det_shape):
    """Generates random events and computes the Mahalanobis distance for each.  
        Returns the list of all sampled M. distance values."""
    # for compatibility with PretrainableModel wrapper class
    if hasattr(model, "full_model"):
        n_tracks = model.n_tracks
        model = model.full_model
    gen = gen_n_tracks(n_tracks=n_tracks, 
            det_shape=det_shape, batch_size=1)
    m = [] # simulated values of M. distance
    for _ in range(ntoys):
        test_data = gen.next()
        test_event = test_data[0][0]
        true = test_data[1][0,:,:2]
        test_pred = model.predict(np.asarray([test_event]))[0]
        pred = test_pred[:,:2]
        cov = cov_from_params( test_pred[:,2:] )
        m += squared_mahalanobis_distance(true, pred, cov).flatten().tolist()
    return m

def make_qq_plot(mahalanobis_vals):
    ref = range(0,15)
    stats.probplot(mahalanobis_vals, dist=stats.chi2, sparams=(2,), fit=False, plot=plt)
    plt.axis([0,10,0,10])
    plt.plot(ref, ref, 'k--')
    plt.show()

def get_layer_by_name(model, layer_name):
    """Finds the layer with specified name.
        model: keras model
        layer_name: name of the layer to return
        Returns: model layer with requested name.
        Throws ValueError if layer is not found."""
    layer = None
    for lyr in model.layers:
        if lyr.name == layer_name:
            layer = lyr
            break
    if layer is None:
        raise ValueError("Requested layer name {} not found".format(
            layer_name))
    return layer

def get_visualization_function(layer, filter_num, input_img):
    """Builds a function that computes the gradient of the 
        desired convolutional filter's activation with respect to
        an input image.
        layer: keras convolutional layer
        filter_num: index of the filter to visualize
        input_img (keras tensor): model input image"""
    loss = K.mean( layer.output[:, filter_num, :, :] )
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    return iterate

def normalize_image(img):
    """Converts image to have mean 0.5 and standard deviation 0.1.
        Clips the image to [0,1]."""
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    img += 0.5
    img = np.clip(img, 0, 1)
    return img

def visualize_filter(model, layer_name, filter_num, 
        det_shape=default_det_shape, n_steps=20, step=1):
    """Finds an image that maximizes the activation of a given
        convolutional filter.
        Inspired by:
        https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

        model: keras model object containing the convolutional layer
        layer_name: string 
        filter_num: int
        det_shape: tuple of ints (detector depth, detector width)
        n_steps: number of steps to run gradient ascent for
        step: learning rate for gradient ascent

        Returns: numpy array representing the visualized filter image"""
    input_img = model.layers[0].input
    layer = get_layer_by_name(model, layer_name)
    iterate = get_visualization_function(layer, filter_num, input_img)
    input_data = np.random.random((1, 1, det_shape[0], det_shape[1]))
    for _ in range(n_steps):
        loss_val, grads_val = iterate([input_data])
        input_data += grads_val * step
    img = normalize_image( input_data[0,0,1:-1,1:-1] )
    return img
