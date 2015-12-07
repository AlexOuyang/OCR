import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pylab import imread, imshow, figure, show, subplot, plot, scatter



def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out



def build_classifier(trainedResult):
    """ Takes in the trained result and returns a digit classifier 
        The returned digit classifier takes in a 28*28 X 1 digit 
        and outputs 10 probabilities corresponding to 10 digits
    """
     # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    network = build_mlp(input_var)

    # load trained weights
    with np.load(trainedResult) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)

    prediction = lasagne.layers.get_output(network, deterministic=True)

    f_output = theano.function([input_var], prediction)

    return f_output


def predict(img, classifier):
    """ Takes in a binary 28 by 28 image matrix  
    returns the digit with highest probability """
    instance = img.reshape(-1, 1, 28, 28) # the classifier takes in a 4d array
    instance = instance[0][None,:,:]
    results = classifier(instance).ravel()
    digit = np.argmax(results)
    probability = float(results[digit])
    return digit, probability



