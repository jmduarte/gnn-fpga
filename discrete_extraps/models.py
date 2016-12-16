"""
This file contains code to construct keras models.
"""

# External imports
from keras import models
from keras import layers

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
    inputs = layers.Input(shape=(length, dim))
    hidden = layers.LSTM(output_dim=hidden_dim, return_sequences=True)(inputs)
    outputs = layers.TimeDistributed(layers.Dense(dim, activation='softmax'))(hidden)
    model = models.Model(input=inputs, output=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_deep_model(length, dim, dim=100,
                     loss='categorical_crossentropy',
                     optimizer='Nadam', metrics=['accuracy']):
    """
    Build the deep LSTM model.
    
    This is a sequence to sequence model with the following architecture:
        Input -> Dense -> LSTM -> Dense -> Dense -> Output
    Input and output data must have shape:
        (num_batch, length, dim).
    """
    inputs = layers.Input(shape=(length, dim))
    hidden1 = layers.TimeDistributed(layers.Dense(dim, activation='relu'))(inputs)
    hidden2 = layers.LSTM(output_dim=dim, return_sequences=True)(hidden1)
    hidden3 = layers.TimeDistributed(layers.Dense(dim, activation='relu'))(hidden2)
    outputs = layers.TimeDistributed(layers.Dense(dim, activation='softmax'))(hidden3)
    model = models.Model(input=inputs, output=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model