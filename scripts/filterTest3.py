import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage import data, img_as_uint, img_as_float
from skimage.filters import roberts, sobel, scharr, prewitt

imageFile = '../pics/beach2.png'
image = data.imread(imageFile, as_grey=True)
edge_roberts = roberts(image)
edge_sobel = sobel(image)

print edge_sobel

fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')

plt.tight_layout()
plt.show()
