import matplotlib.pyplot as plt
import numpy as np
import ocr

from pylab import imread, imshow, figure, show, subplot, plot, scatter
from scipy.cluster.vq import kmeans, vq
from skimage import data, img_as_uint, img_as_float
from skimage.external.tifffile import imsave
from skimage.filters import threshold_otsu, threshold_adaptive, threshold_yen
from skimage.segmentation import clear_border

imageFile = '../pics/14.png'
image = imread(imageFile)
img = data.imread(imageFile, as_grey=True)

global_thresh = threshold_yen(img)
# True False binary matrix represent color value of the img using global thresholding
binary_global = img > global_thresh

block_size = 40

# True False binary matrix represent color value of the img using adaptive thresholding
binary_adaptive = threshold_adaptive(img, block_size, offset=0)

# 0 1 binary matrix
img_bin_global = clear_border(img_as_uint(binary_global))

# 0 1 binary matrix 
img_bin_adaptive = clear_border(img_as_uint(binary_adaptive))


bin_pos_mat = ocr.binary_matrix_to_position(binary_adaptive)

np.savetxt("test.txt",bin_pos_mat) # %.5f specifies 5 decimal round

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
figure(1)
subplot(311)
imshow(image)
subplot(312)
scatter(bin_pos_mat[:,1], bin_pos_mat[:,0])
# imshow(np.flipud(clustered))
subplot(313)
imshow(binary_adaptive)

show()



# plt.gray()

# ax0.imshow(image)
# ax0.set_title('Image')

# ax1.imshow(binary_global)
# ax1.set_title('Global thresholding')

# ax2.imshow(binary_adaptive)
# ax2.set_title('Adaptive thresholding')



# np.savetxt("test.txt", img_bin_adaptive)

# imsave('result.tif', img_bin_adaptive)

# for ax in axes:
#     ax.axis('off')

# plt.show()