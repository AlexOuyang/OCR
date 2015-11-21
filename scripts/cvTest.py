import cv2
import numpy as np
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter, title
import ocr

# img = cv2.imread('../pics/five.png', 0)
# saved_image_name = '../pics/cropped/cvPic1.png'
# blur = cv2.GaussianBlur(img,(5,5),0)
# edge = cv2.Canny(blur,50,50)
# # ret,thresh = cv2.threshold(edge,127,255,cv2.THRESH_OTSU)
# ret,thresh = cv2.threshold(edge,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
# imshow(blur)
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
    cropped_img = ocr.crop_digit(img_name, 3)
    cv2.imwrite(saved_image_name, cropped_img)

    idx = 330 + i
    subplot(idx)
    imshow(cropped_img)

show()

figure(2)
title("Pretty Print")
cropped_img = ocr.crop_digit('../pics/print.png', 3)
cv2.imwrite('../pics/cropped/cvPic1.png', cropped_img)
subplot(111)
imshow(cropped_img)
show()

figure(3)
title("handWriting")
cropped_img = ocr.crop_digit('../pics/handWriting.jpg', 3)
cv2.imwrite('../pics/cropped/cvPic2.png', cropped_img)
subplot(111)
imshow(cropped_img)
show()


figure(4)
title("handWriting")
cropped_img = ocr.crop_digit('../pics/lotsOfDigits.png', 3)
cv2.imwrite('../pics/cropped/cvPic3.png', cropped_img)
subplot(111)
imshow(cropped_img)
show()







