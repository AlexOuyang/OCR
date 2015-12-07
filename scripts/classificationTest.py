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
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter, title
import multilayerPerceptron as mlp


f_output = mlp.build_classifier('../trainedResult/model.npz')

input_img = imread('../pics/cropped/0.png')

print mlp.predict(input_img, f_output)


plt.subplot(211)
plt.imshow(input_img, cmap=cm.binary)
instance = input_img.reshape(-1, 1, 28, 28)
pred = f_output(instance)
N = pred.shape[1]
plt.subplot(212)
#result with probability
plt.bar(range(N), pred.ravel())

plt.show()

