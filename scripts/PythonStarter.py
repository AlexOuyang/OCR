
# coding: utf-8

# In[ ]:

###### PYTHON STARTER CODE ######
# RUN YOUR DATA ON ONE OF THE TWO EXAMPLE PIPELINES BELOW, DEPENDING ON WHETHER
# YOU WANT TO DO SUPERVISED (regression) OR UNSUPERVISED LEARNING (kmeans)
#
# things you need to do (ONLY ONE PER TEAM):
#  - download the following packages if you do not have them: numpy, scipy, matplotlib
#  - import your data into python and name the variables appropriately
#  - run one of the two processes, the easiest thing to do is to run this script in
#       iPython notebook (this is an useful thing to learn). but if you want, you could
#       copy and paste the relevant code snippet into your own python script
#  - produce the plots and change the x/y axis labels and title to reflect your data
#  - email me your plot and a short description of what the data you plotted represent
#       i.e. X is height of all elephants I measured, and Y is their age, etc. or
#       X[0] is the height of all elephants I measured, and X[1] is their weight
#
# the computations perform themselves, the point of this is to check in and see that
# you've been able to enter your data at the very least


# In[ ]:

### LINEAR REGRESSION\n",

# #import packages
# get_ipython().magic(u'matplotlib inline')
# import numpy as np
# import scipy as sp
# import matplotlib.pyplot as pp

# # ENTER YOUR OWN DATA HERE:
# # if you're doing supervised learning, X should be the values of a
# # single dimension (feature) for all your data points, and Y should
# # be your prediction (output) value

# X = [float(l.strip().split()[0]) for l in open("scaledfaithful.txt")]
# Y = [float(l.strip().split()[1]) for l in open("scaledfaithful.txt")]
# #__________________________

# # only want the first 100 examples
# X = X[:100]
# Y = Y[:100]

# # we append 1's to the data vector because linear regression requires an offset
# featX = [[1.,i] for i in X]

# # get line of best fit (offset & slope) by running least squares regression
# theta = np.linalg.lstsq(featX, Y)[0]

# # plotting
# pp.plot(X,Y,'o')
# pp.plot([min(X), max(X)], np.array([min(X), max(X)])*theta[1]+theta[0])

# #### CHANGE THESE LABELS TO MATCH YOUR DATA
# pp.title('Linear Regression')
# pp.xlabel('Feature 1')
# pp.ylabel('Predictor')
# pp.show()


# In[ ]:

##### K-MEANS

#import packages
# get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy.cluster.vq as CL
import matplotlib.pyplot as pp

from pylab import imread, imshow, figure, show, subplot
from scipy.cluster.vq import kmeans, vq

img = imread('../pics/1.png')


# ENTER YOUR OWN DATA HERE:
# if you're doing unsupervised learning, X should be the values of
# 2 dimensions (feature) for all your data points

X = [[float(l.strip().split()[0]),float(l.strip().split()[1])] for l in open("scaledfaithful.txt")]

# reshaping the pixels matrix to read in for k means
X = np.reshape(img, (img.shape[0] * img.shape[1], 3))
#________________________________

# take first 100 points and cluster into numK centroids
# you can change numK to see how it changes the results
X = X[:100]
numK = 2
centroids,sq = CL.kmeans(np.array(X), numK)

#compute distance of each sample to the centroids
dist = [(np.sum((np.array(X) - np.array([centroids[i]]))**2,1)) for i in range(numK)]

#assign sample to closest cluster
class_assigned = [(np.where(np.transpose(dist)[i]==min(np.transpose(dist)[i])))[0][0] for i in range(100)]
clustered = [[X[i] for i, clu in enumerate(class_assigned) if clu == k*1.] for k in range(numK)]

# plotting each class
for k in range(numK):
    pp.plot(np.transpose(clustered[k])[0], np.transpose(clustered[k])[1],'o')

#### CHANGE THESE LABELS TO MATCH YOUR DATA
pp.title('K Means')
pp.xlabel('Feature 1')
pp.ylabel('Feature 2')
pp.show()


# In[ ]:



