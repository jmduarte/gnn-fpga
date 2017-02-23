"""
This file contains code to construct keras models.
"""

# External imports
import numpy as np
from keras.models import Model
from keras.layers import (Input, LSTM, Dense, TimeDistributed, Bidirectional,
                          Conv3D, MaxPooling3D, Dropout, Reshape, Permute, Activation)
from keras.regularizers import l2

def SeqDense(name=None, *args, **kwargs):
    """Shorthand for TimeDistributed Dense layer"""
    return TimeDistributed(Dense(*args, **kwargs), name=name)

def build_lstm_model(length, dim, hidden_dim=100,
                     loss='categorical_crossentropy',
                     optimizer='Nadam', metrics=['accuracy']):
    """
    Build the simple LSTM model.

    This is a sequence to sequence model with the following
    architecture:
        Input -> LSTM -> Dense -> Output
    Input and output data must have shape:
        (num_batch, length, dim).
    """
    inputs = Input(shape=(length, dim))
    hidden = LSTM(output_dim=hidden_dim, return_sequences=True)(inputs)
    outputs = TimeDistributed(Dense(dim, activation='softmax'))(hidden)
    model = Model(input=inputs, output=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_deep_lstm_model(length, dim, hidden_dim=100,
                          loss='categorical_crossentropy',
                          l2reg=0., dropout=0., optimizer='Nadam',
                          metrics=['accuracy']):
    """
    Build the deep LSTM model.

    This is a sequence to sequence model with the following architecture:
        Input -> Dense -> LSTM -> Dense -> Dense -> Output
    Input and output data must have shape:
        (num_batch, length, dim).
    """
    inputs = Input(shape=(length, dim))
    hidden1 = TimeDistributed(
        Dense(hidden_dim, activation='relu', W_regularizer=l2(l2reg)))(inputs)
    hidden2 = LSTM(output_dim=hidden_dim, return_sequences=True,
                   dropout_W=dropout, dropout_U=dropout)(hidden1)
    hidden3 = TimeDistributed(
        Dense(hidden_dim, activation='relu', W_regularizer=l2(l2reg)))(hidden2)
    outputs = TimeDistributed(
        Dense(dim, activation='softmax', W_regularizer=l2(l2reg)))(hidden3)
    model = Model(input=inputs, output=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_bilstm_model(length, dim, hidden_dim=100,
                       loss='categorical_crossentropy',
                       optimizer='Nadam', metrics=['accuracy']):
    """
    Build the bidirectional LSTM model.

    This is a sequence to sequence model with the following
    architecture:
        Input -> BiLSTM -> Dense -> Output
    Input and output data must have shape:
        (num_batch, length, dim).
    """
    inputs = Input(shape=(length, dim))
    hidden = Bidirectional(
        LSTM(hidden_dim, return_sequences=True))(inputs)
    hidden = TimeDistributed(
        Dense(hidden_dim, activation='relu'))(hidden)
    outputs = TimeDistributed(
        Dense(dim, activation='softmax'))(hidden)
    model = Model(input=inputs, output=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_conv_model(shape):
    """
    Build the simple convolutional model.

    This model performs 10 convolutions of size 8x3x3x3, followed by
    a single 3x3x3 filter with a softmax activation on each detector plane.

    Input data shape:
        (num_batch, det_depth, det_width, det_width)
    Output data shape:
        (num_batch, det_depth, det_width*det_width)
    """
    from keras.regularizers import l2
    inputs = Input(shape=shape)
    # Need a 'channel' dimension for 3D convolution, though we have only 1 channel
    hidden = Reshape((1,)+shape)(inputs)
    # 3D convolutional layers
    conv_args = dict(border_mode='same', activation='relu')
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    # Final convolution without activation
    hidden = Conv3D(1, 3, 3, 3, border_mode='same')(hidden)
    # Reshape to flatten each detector layer
    hidden = Reshape((shape[0], shape[1]*shape[2]))(hidden)
    # Final softmax activation
    outputs = TimeDistributed(Activation('softmax'))(hidden)
    # Compile the model
    model = Model(input=inputs, output=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model

def build_convae_model(shape, dropout=0, l2reg=0, pool=(1,2,2)):
    """
    Build the convolutional autoencoder model.

    This model performs several 3D convolutions, increasing channel depth
    while downsampling. After the bottleneck, a fully-connected decoder
    with softmax activation is applied to each detector plane. At the time
    of writing this model, it is not possible to do 3D transpose convolutions
    (deconv3d) with native keras network layers.

    Input data shape:
        (num_batch, det_depth, det_width, det_width)
    Output data shape:
        (num_batch, det_depth, det_width*det_width)
    """
    inputs = Input(shape=shape)
    # Need a 'channel' dimension for 3D convolution,
    # though we have only 1 channel initially.
    hidden = Reshape((1,)+shape)(inputs)
    # 3D convolutional layers
    conv_args = dict(border_mode='same', activation='relu')
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(8, 3, 3, 3, **conv_args)(hidden)
    hidden = MaxPooling3D(pool_size=pool)(hidden)
    hidden = Dropout(dropout)(hidden)
    hidden = Conv3D(16, 3, 3, 3, **conv_args)(hidden)
    hidden = Conv3D(16, 3, 3, 3, **conv_args)(hidden)
    hidden = MaxPooling3D(pool_size=pool)(hidden)
    hidden = Dropout(dropout)(hidden)
    hidden = Conv3D(32, 3, 3, 3, **conv_args)(hidden)
    hidden = MaxPooling3D(pool_size=pool)(hidden)
    hidden = Dropout(dropout)(hidden)
    hidden = Conv3D(64, 3, 3, 3, **conv_args)(hidden)
    hidden = MaxPooling3D(pool_size=pool)(hidden)
    hidden = Dropout(dropout)(hidden)
    hidden = Conv3D(96, 3, 2, 2, **conv_args)(hidden)
    hidden = MaxPooling3D(pool_size=pool)(hidden)
    hidden = Dropout(dropout)(hidden)
    hidden = Conv3D(128, 3, 1, 1, **conv_args)(hidden)
    # Permute dimensions to group detector layers:
    # (channels, det_layers, w, w) -> (det_layers, channels, w, w)
    PermuteLayer = Permute((2, 1, 3, 4))
    hidden = PermuteLayer(hidden)
    # Reshape to flatten each detector layer: (det_layers, -1)
    perm_shape = PermuteLayer.output_shape
    flat_shape = (perm_shape[1], np.prod(perm_shape[2:]))
    hidden = Reshape(flat_shape)(hidden)
    # Output softmax
    outputs = TimeDistributed(
        Dense(shape[1]*shape[2], activation='softmax',
              W_regularizer=l2(l2reg)))(hidden)
    # Compile the model
    model = Model(input=inputs, output=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam', metrics=['accuracy'])
    return model
