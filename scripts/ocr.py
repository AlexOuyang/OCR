import matplotlib.pyplot as plt
import numpy as np

from pylab import imread, imshow, figure, show, subplot, plot, scatter
from scipy.cluster.vq import kmeans, vq
# from skimage import data, img_as_uint, img_as_float
# from skimage.external.tifffile import imsave
# from skimage.filters import threshold_otsu, threshold_adaptive, threshold_yen
# from skimage.segmentation import clear_border


def img_kmeans(imgFile, k):
    img = imread(imgFile)
    # reshaping the pixels matrix to read in for k means
    pixel = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # performing the clustering on shaped pixels
    centroids, _ = kmeans(pixel, k)      # centeroids contains the 2 representative color of the image

    # quantization of 2 for color
    qnt, _ = vq(pixel, centroids)

    # reshaping the result of the quantization, clustered contains the cluster numbers
    clustered = np.reshape(qnt, (img.shape[0], img.shape[1]))   
    clustered_color = centroids[clustered]   #color the clustered
    return clustered, clustered_color


def binary_matrix_to_position(binMat):
    """ takes in n x n binary True False color matrix and output a 
        n*n x 1 position matrix represent the position of colored digits 
    """
    binPosMat = np.empty((0,2), int)
    # go through the clustered matrix and get the position of the pixel that's not white
    for (x,y), pixel in np.ndenumerate(binMat):
        pixelColor =  binMat[x,y]
        if pixelColor == False:
            binPos = np.empty((1,2), int)
            binPos[0,0] = x
            binPos[0,1] = y
            binPosMat = np.append(binPosMat, binPos, axis=0)

    # flip the matrix
    return binPosMat


def binary_matrix_to_position_num(binMat):
    """ takes in n x n binary True False color matrix and output a 
        n*n x 1 position matrix represent the position of colored digits 
    """
    binPosMat = np.empty((0,2), int)
    # go through the clustered matrix and get the position of the pixel that's not white
    for (x,y), pixel in np.ndenumerate(binMat):
        pixelColor =  binMat[x,y]
        if pixelColor > 0:
            binPos = np.empty((1,2), int)
            binPos[0,0] = x
            binPos[0,1] = y
            binPosMat = np.append(binPosMat, binPos, axis=0)

    # flip the matrix
    return binPosMat

def binary_matrix_to_position_2(clustered):
    """ we assume the first color pixel that's most upper left of the image is not the 
    color of the number, a quick fix 
    """
    not_digit_color = clustered[1,1]

    binPosMat = np.empty((0,2), int)
    # go through the clustered matrix and get the position of the pixel that represent a digit
    for (x,y), pixel in np.ndenumerate(clustered):
        pixelColor =  clustered[x,y]
        if pixelColor != not_digit_color:
            binPos = np.empty((1,2), int)
            binPos[0,0] = x
            binPos[0,1] = y
            binPosMat = np.append(binPosMat, binPos, axis=0)

    return binPosMat
        
