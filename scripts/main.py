""" 
===================================================
    Digit Classification Using MLP and OpenCV 
===================================================

Summary:
        Digit Segmentation and Classification Using 
        MLP and OpenCV on GoogleStreetView Data Base

        Created by:  Chenxing Ouyang & Jiali Xie

"""
import sys
import cv2
import numpy as np
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter, title
import ocr
import multilayerPerceptron as mlp

print(__doc__)

ocr.delete_files("../pics/cropped/")


print ("Preprocessing Image, Cropping Digits Into 28 X 28 Image Matrices\n")
#  save_digit_to_binary_img_as_mnist(imgName, saveToFile = True, imgSize = 100, boundingRectMinSize = 5)
cropped_img_for_show, cropped_digits = ocr.save_digit_to_binary_img_as_mnist("../pics/12.png",saveToFile = True)

print ("Image Preprocessing Done, %d Potential Digits Were Cropped Out\n" % len(cropped_digits))


print ("Building Multilayer Perceptron Network From Trained Model\n")
mlp_classifier = mlp.build_classifier('../trainedResult/model.npz')

# input_img = imread('../pics/cropped/0.png')
# print mlp.predict(input_img, mlp_classifier)

print ("Predicting Results\n")
print ("Image    Digit     probability")

# Loading from matrix
# index = 0
# for input_digit in cropped_digits:
#     digit, probability = mlp.predict(input_digit, mlp_classifier)
#     print ("%d.png      %d         %f" % (index, digit, probability))
#     index += 1

# Loading from Image
index = 0
for input_digit in cropped_digits:
    path = "../pics/cropped/" + str(index) + ".png"
    input_digit = imread(path)
    digit, probability = mlp.predict(input_digit, mlp_classifier)
    print ("%d.png      %d         %f" % (index, digit, probability))
    index += 1



figure(1)
title("handWriting")
cv2.imwrite("../pics/cropped/cvPic3.png", cropped_img_for_show)
subplot(111)
imshow(cropped_img_for_show)
show()
