import numpy as np

import keras.backend as K
from keras.regularizers import Regularizer
from keras.layers.core import Layer

from theano.printing import Print

def stepwise_kl_divergence(y):
    """Input: tensor of shape (N_obs, N_timesteps, N)
              in which axis 2 represents a probability distribution
              at each time step
       Output: tensor of shape (N_obs, N_timesteps-1)
              in which the nth entry along axis 1 is the KL divergence
              of the (n+1)th row of the input from the nth row"""
    y = K.clip(y, K.epsilon(), 1)
    y1 = y[:,1:]
    y2 = y[:,:-1]
    kl =  K.sum(y1 * K.log(y1 / y2), axis=2)
    return kl

class KLRegularizer(Regularizer):
    """This regularizer assumes that dimension 1 in the target layer
       indexes time steps.  For each time step t>0, the regularizer applies
       a penalty proportional to the KL divergence from time step t-1."""

    def __init__(self, k):
        self.k = K.cast_to_floatx(k)
        self.uses_learning_phase = True
        self.layer = None

    def set_layer(self, layer):
        if self.layer is not None:
            raise Exception('Regularizers cannot be reused')
        self.layer = layer

    def __call__(self, loss):
        if self.layer is None:
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_input_at(i)
            regularized_loss += K.sum( self.k * stepwise_kl_divergence(output) )
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'k': float(self.k)}

class KLRegularization(Layer):
    '''Layer that passes through its input unchanged, but applies an update
    to the cost function based on the KL divergence between adjacent time steps.

    # Arguments
        k: regularization factor (positive float).

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, k=0., **kwargs):
        self.supports_masking = True
        self.k = k

        super(KLRegularization, self).__init__(**kwargs)
        kl_regularizer = KLRegularizer(k=k)
        kl_regularizer.set_layer(self)
        self.regularizers = [kl_regularizer]

    def get_config(self):
        config = {'k': self.k}
        base_config = super(KLRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
