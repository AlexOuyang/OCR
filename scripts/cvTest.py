import cv2
import numpy as np
from cv2 import THRESH_OTSU
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter, title
import ocr

# img = cv2.imread('../pics/1.png')
# saved_image_name = '../pics/cropped/cvPic1.png'
# edge = cv2.Canny(img,50,50)
# ret,thresh = cv2.threshold(edge,127,255,THRESH_OTSU)
# # gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# idx =0 
# boundingRectMinSize = 10
# for cnt in contours:
#     idx += 1
#     if cnt.shape[0] >= boundingRectMinSize:
#         x,y,w,h = cv2.boundingRect(cnt)
#         # roi=im[y:y+h,x:x+w]
#         cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),1)
#         cv2.rectangle(thresh,(x,y),(x+w,y+h),(200,0,0),1)
#         # cv2.imwrite('../pics/cropped/' + str(idx) + '.png', im)
#         # im = cv2.imread('../pics/beach2.png')


# cropped_img = img
# cv2.imwrite(saved_image_name, cropped_img)

# # cv2.imshow('img',im)
# # cv2.waitKey(5000)
# # cv2.destroyAllWindows()


# figure(1)
# subplot(221)
# title('Original Image')
# imshow(img)
# subplot(222)
# title('Canny Edge Detection')
# imshow(edge)
# subplot(223)
# title('Bounding Box on Otsu Threshold')
# imshow(thresh)
# subplot(224)
# title('Bounding Box on original image')
# imshow(cropped_img)
# show()

figure(1)

for i in range(1,10):
    img_name = '../pics/' + str(i) + '.png'
    saved_image_name = '../pics/cropped/cropped_' + str(i) + '.png'
    cropped_img = ocr.crop_digit(img_name, saved_image_name, 10)
    cv2.imwrite(saved_image_name, cropped_img)

    idx = 330 + i
    subplot(idx)
    imshow(cropped_img)

show()












