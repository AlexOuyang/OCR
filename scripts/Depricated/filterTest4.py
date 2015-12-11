import cv2
from cv2 import THRESH_OTSU
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../pics/14.png',0)

img = cv2.Canny(img,50,50)

ret,thresh = cv2.threshold(img,127,255,THRESH_OTSU)

contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
# print cnt
M = cv2.moments(cnt)
# print M

x,y,w,h = cv2.boundingRect(cnt)
# print x,y,w,h

# cv2.rectangle(thresh,(0,0),(35+30,45+30),(255,0,0),5)
# cv2.line(thresh,(0,0),(511,511),(255,0,0),5)


cv2.imshow('rectangle',thresh)
# wait for 5 seconds then destroy window
cv2.waitKey(5000)
cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img,cmap = 'gray')

# plt.subplot(122),plt.imshow(thresh,cmap = 'gray')

# plt.show()

cv2.imwrite('../pics/result.png', thresh)
# edges = cv2.Canny(img,100,200)

# cv2.imshow('rectangle',thresh)
# # wait for 5 seconds then destroy window
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')

#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()

