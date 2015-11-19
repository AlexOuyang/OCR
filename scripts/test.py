
# from PIL import Image

# Image.open('../pics/1.png').convert('L').save('../pics/1_gray.png')

import numpy as np
from pylab import imread, imshow, figure, show, subplot
from scipy.cluster.vq import kmeans, vq

img = imread('../pics/1.png')

# reshaping the pixels matrix to read in for k means
pixel = np.reshape(img, (img.shape[0] * img.shape[1], 3))

# performing the clustering on shaped pixels
centroids, _ = kmeans(pixel, 2)      # centeroids contains the 2 representative color of the image

# quantization of 2 for color
qnt, _ = vq(pixel, centroids)

# reshaping the result of the quantization, clustered contains the cluster numbers
clustered = np.reshape(qnt, (img.shape[0], img.shape[1]))   
print clustered
# clustered = centroids[clustered]   #color the clustered


figure(1)
subplot(211)
imshow(img)
subplot(212)
imshow(clustered)
# imshow(np.flipud(clustered))
show()
