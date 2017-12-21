################################################################################
Here are notebooks used to train neural networks and extract data.

################################################################################
ACTS-EXTRACT.ipynb

"How data was extracted from ACTS-generated events. Data had to be parsed,
cleaned and organized to transition the text files into useable python pandas
data frames. Data was retrieved from the Caltech cluster under the directory,
/inputdata/ACTS."

################################################################################
ACTS-MU10-PT1000-T50.ipynb
Train is list of 52325 events.
Test is list of 13082 events.
Tracks per event ranging within [1, 50].
Tracks per event distribution is Poisson with mu=10 and clipped at 50.
No noise (every hit belongs to some particle's track)
3 Bidirectional GRU each with 256 units.
Input matrices must have shape (200, 3). Padding is used to regulate this.
Percent of hits assigned correctly: 97.61%
Percent of tracks assigned correctly: 96.14%
Percent of events with the correct number of tracks: 95.27%

"How a neural network with 3 Bidirectional GRU layers was created, fitted and
evaluated.

The frames used to train and test the model were extracted using the
ACTS-EXTRACT.ipynb notebook.

Tracks tended to not cross often, since the
momentums were high. Execution cell 28 illustrates momentums as histograms.

There were only 4 layers within the detector, so each track had
at most 4 hits. Misclassifications were usually the result of the network
having difficulty distinguishing between two positionally close hits. Perhaps
with more detector layers, the network can better trace out tracks.

As the number of tracks increase within an event, the detector has a more
difficult time classifying the hits to tracks."

################################################################################
UNIF-10N-25T-25000E.ipynb
Train is list of 25000 events
Test is list of 3600 events
Tracks per event ranging within [1, 25].
For the training set, tracks per event distribution is uniform, such that for
each n within [1, 25], there are 1000 events with n tracks.
For the testing set, tracks per event distribution is Poisson distributed with
mu=12 and clipped at 22.
Each event contains 10 noisy hits (randomly inserted hits that belong to no
particle's track).
3 Bidirectional GRU each with 256 units.
Input matrices must have shape (235, 3). Padding is used to regulate this.
Percent of hits assigned correctly: 84.47%
Percent of tracks assigned correctly: 59.62%
Percent of events with the correct number of tracks: 83.28%

"How a neural network with 3 Bidirectional GRU layers was created, fitted and
evaluated.

The data that was used to test the model comes from the RAMP set. In order to
make the set 3-dimensional, a randomly-sloped linear component was added to
each track. For a given track, each hit's z component was equal to the hit's
radius multiplied by a randomly selected constant slope value.

Train data was extracted from the RAMP set in the following way:
1. Collect all tracks from the RAMP set.
2. For n within [1, 25], create 1000 events with n tracks.
3. Tracks within an event were sampled from the collection of tracks.
4. Tracks chosen were then randomly rotated about the detector's origin.

The code used can be found in the generator directory."

################################################################################
-HOW-TO-USE-.ipynb

"A walkthrough for how to use a notebook to train a network with prepared data."


################################################################################
CUSTOM-LOSS-PRACTICE-CAT.ipynb

"A file used to practice custom loss functions with toy data and using a 
categorical output framework. The results were that custom loss functions
are compatible with categorical output representations, but it takes significantly
more epochs to train them. The custom loss function is special in that it makes
reference to the input tensor, the output prediction and the output truth.
Normally, loss functions only reference the prediction and the truth."

################################################################################
CUSTOM-LOSS-PRACTICE-MSE.ipynb

"A file used to practice custom loss functions with toy data and using mean
square error. The results were that loss functions that reference the input
tensor are possible."

################################################################################
CUSTOM-LOSS-PRACTICE-RNN.ipynb

"A file used to practice custom loss functions with toy data and using RNN
neural network layers (particularly GRU). The results were that the methods
I used to test RNN are not compatible with custom loss functions that
reference the input tensor."

################################################################################
CUSTOMLoss.ipynb

"A file used to test a custom loss function that evaluates the linear regression
of each track within an event. The data used is the same as
ACTS-MU10-PT1000-T50.ipynb. The results were that the custom loss function
does not perform well and was difficult to construct for the tracking problem." 
