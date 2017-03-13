# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".

from __future__ import division

import numpy as np

import keras.backend as K
from keras.layers.recurrent import LSTM

class Attention(LSTM):
    """LSTM with an additional input: 'annotation' vectors
        describing position-dependent information about an
        input image. The input tensor should be of shape
        (batch_size, num_annotations, annotation_dim).
        Attributes:
        -context_dim: dimension of the annotation vectors
        -annot: annotation vectors for the current event
    """

    def call(self, x, mask=None):
        if isinstance(x, (tuple, list)):
            # It's a bit hacky to make the annotation vector a 
            # class attribute. But since step() has a fixed function
            # signature this is needed to pass the vector in
            x, self.annot = x
            mask = mask[0]
        else:
            raise TypeError(("Attention layer must have two inputs:"
                             "LSTM input and annotations"))
        return super(Attention, self).call(x, mask)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask[0]
        else:
            return None

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0][0], input_shape[0][1], 
                    self.output_dim)
        else:
            return (input_shape[0][0], self.output_dim)

    def get_initial_states(self, x):
        """LSTM initial state is determined by the mean of
            all annotation vectors via an MLP"""
        mean_annot = K.mean(self.annot, axis=1)

        h = K.tanh( K.dot(mean_annot, self.W_h) + self.b_h )
        c = K.tanh( K.dot(mean_annot, self.W_c) + self.b_c )

        return [h, c]

    def build(self, input_shape):
        """Input shape must be a list of two items:
            -LSTM input shape
            -Annotation vector shape
           Weights are the same as for an LSTM, with a few extras:
            -A set of weights multiplying the context vector in the 
                calculation of i/f/o/g
            -A MLP for the calculation of the context vector
        """
        if isinstance(input_shape, list) and len(input_shape) > 1:
            lstm_shape, annot_shape = input_shape
            self.num_annots = annot_shape[1]
            self.context_dim = annot_shape[2]
        else:
            raise TypeError(("Attention layer must have two inputs: "
                             "LSTM input and annotations"))
        super(Attention, self).build(lstm_shape)

        if self.consume_less != 'gpu':
            raise NotImplementedError("Only consume_less = 'gpu' is supported")

        self.W_att = self.add_weight((self.context_dim, 4 * self.output_dim),
                                 initializer=self.init,
                                 name='{}_W_att'.format(self.name),
                                 regularizer=None)

        # MLP weights for computing context vector
        self.Wc_att = self.add_weight((self.context_dim, self.context_dim),
                                 initializer=self.init,
                                 name='{}_Wc_att'.format(self.name),
                                 regularizer=None)
        self.Uc_att = self.add_weight((self.output_dim, self.context_dim),
                                 initializer=self.init,
                                 name='{}_Uc_att'.format(self.name),
                                 regularizer=None)
        self.bc_att = self.add_weight((self.context_dim,),
                                 initializer=self.init,
                                 name='{}_bc_att'.format(self.name),
                                 regularizer=None)
        self.U_att = self.add_weight((self.context_dim, 1),
                                 initializer=self.init,
                                 name='{}_U_att'.format(self.name),
                                 regularizer=None)

        # MLP weights for computing initial LSTM state
        self.W_h = self.add_weight((self.context_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W_h'.format(self.name),
                                 regularizer=None)
        self.b_h = self.add_weight((self.output_dim,),
                                 initializer=self.init,
                                 name='{}_b_h'.format(self.name),
                                 regularizer=None)
        self.W_c = self.add_weight((self.context_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W_h'.format(self.name),
                                 regularizer=None)
        self.b_c = self.add_weight((self.output_dim,),
                                 initializer=self.init,
                                 name='{}_b_c'.format(self.name),
                                 regularizer=None)

    def step(self, x, states):
        """Invoked by parent class call() method"""
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        context = self.compute_context(self.annot, h_tm1)

        # The context vector feeds into the LSTM computation here
        z = ( K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U)
                + K.dot(context, self.W_att) + self.b )

        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim: 2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)

        h = o * self.activation(c)
        return h, [h, c]

    def compute_context(self, features, hidden):
        """Computes the context vector via Eqs. 4-6 in the paper"""
        e = K.relu( K.dot(features, self.Wc_att)
                + K.expand_dims(K.dot(hidden, self.Uc_att), 1) + self.bc_att )
        e = K.squeeze( K.dot(e, self.U_att), axis=-1 )
        alpha = K.softmax(e)
        context = K.sum(K.expand_dims(alpha) * features, axis=1)
        return context
