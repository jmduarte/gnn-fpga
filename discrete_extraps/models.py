"""
This file contains code to construct keras models.
"""

# External imports
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional
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
