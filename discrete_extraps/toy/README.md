# Discrete detector track extrapolation and hit classification

This area contains notebooks and code for track-finding methods based
on discrete detector toy data and track extrapolations or hit classifications.

## Notebooks

These notebooks demonstrate full execution of the models.

* [LSTM_Toy.ipynb](LSTM_Toy.ipynb) - LSTM for current-layer predictions in the
  3D detector toy data.

* [LSTM_Toy_KF.ipynb](LSTM_Toy_KF.ipynb) - LSTM for next-layer predictions in
  the 3D detector toy data. This model functions similar to a Kalman Filter.

* [LSTM_Toy_MultiTarget.ipynb](LSTM_Toy_MultiTarget.ipynb) - LSTM model which
  has an additional output target which predicts track parameters.

* [Conv3D_Toy.ipynb](Conv3D_Toy.ipynb) - 3D convolutional network for hit
  classification/assignment in the 3D detector toy data. This model architecture
  uses a bottleneck similar to an autoencoder.

* [Conv3D_Toy_2.ipynb](Conv3D_Toy_2.ipynb) - An alternate, simpler 3D conv
  model with constant size convolutional layers.

* [DataGen1D.ipynb](DataGen1D.ipynb) - A notebook used to develop data
  generation code for the 2D toy detector data (with 1D detector layers).

* [DataGen2D.ipynb](DataGen2D.ipynb) - A notebook used to develop data
  generation code for the 3D toy detector data (with 2D detector layers).

## Modules

* [toydata.py](toydata.py) - Contains the code to generate toy data consisting
  of straight line tracks in a 3D detector composed of 2D square plane layers.

* [metrics.py](metrics.py) - Contains some helper code for metric evaluation.

* [models.py](models.py) - Definitions of some of the common models utilized in
  the notebooks.

* [drawing.py](drawing.py) - Utility code for plotting data with matplotlib.
