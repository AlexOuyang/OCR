
# from PIL import Image

# Image.open('../pics/1.png').convert('L').save('../pics/1_gray.png')

import numpy as np
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter
from scipy.cluster.vq import kmeans, vq
import ocr
import matplotlib.pyplot as plt
import numpy as np

from pylab import imread, imshow, figure, show, subplot, plot, scatter
from scipy.cluster.vq import kmeans, vq
from skimage import data, img_as_uint, img_as_float
from skimage.external.tifffile import imsave
from skimage.filters import threshold_otsu, threshold_adaptive, threshold_yen
from skimage.segmentation import clear_border


img = '../pics/14.png'
# img = imread('../pics/1.png')

image = imread(img)

clustered, clustered_color = ocr.img_kmeans(img, 2)

binPosMat = ocr.binary_matrix_to_position_2(clustered)

# color filtered using k means
# imsave("../pics/pic_clustered.png", clustered_color)


figure(1)
subplot(221)
imshow(image)
subplot(222)
imshow(clustered_color)
subplot(223)
scatter(binPosMat[:,0], binPosMat[:,1])
subplot(224)
show()