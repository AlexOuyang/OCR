import matplotlib.pyplot as plt
import numpy as np

from pylab import imread, imshow, figure, show, subplot
from skimage import data, img_as_uint, img_as_float
from skimage.external.tifffile import imsave
from skimage.filters import threshold_otsu, threshold_adaptive, threshold_yen


#load image
image = data.imread('../pics/5.png', as_grey=True)


global_thresh = threshold_yen(image)
binary_global = image > global_thresh

block_size = 40


binary_adaptive = threshold_adaptive(image, block_size, offset=0)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

result = img_as_uint(binary_adaptive)

#np.savetxt("text.txt", result)

imsave('result.tif', result)

for ax in axes:
    ax.axis('off')

plt.show()

