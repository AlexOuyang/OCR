""" 
===================================================
    Digit Classification Using MLP and OpenCV 
===================================================

Summary:
        Digit Segmentation and Classification Using 
        MLP and OpenCV on GoogleStreetView Data Base

        Created by:  Chenxing Ouyang & Jiali Xie

Usage:
        Press 'c' to take a screen shot for analysis
        Press 'q' or 'ESC' to quit


"""
import sys
import cv2
import numpy as np
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter, title
import ocr
import multilayerPerceptron as mlp

print(__doc__)


pause = False


print ("Building Multilayer Perceptron Network From Trained Model\n")
mlp_classifier = mlp.build_classifier('../trainedResult/model.npz')


def analyze_digit (img):
    """ Takes in an image matrix, crops out the digits and outputs it to file """

    ocr.delete_files("../pics/cropped/")

    print ("Preprocessing Image, Cropping Digits Into 28 X 28 Image Matrices\n")
    cropped_img_to_show, cropped_thresh_to_Show, cropped_digits = ocr.save_digit_to_binary_img_as_mnist(img,saveToFile = True, imgSize = 100)

    print ("Image Preprocessing Done, %d Potential Digits Were Cropped Out\n" % len(cropped_digits))

    # print ("Predicting Results\n")
    # print ("Image    Digit     probability")


    index = 0
    for input_digit in cropped_digits:
        path = "../pics/cropped/" + str(index) + ".png"
        input_digit = imread(path)
        digit, probability = mlp.predict(input_digit, mlp_classifier)
        print ("%d.png      %d         %f" % (index, digit, probability))
        index += 1

    # figure(1)
    # title("handWriting")
    # cv2.imwrite("../pics/cropped/cvPic3.png", cropped_img_to_show)
    # subplot(111)
    # imshow(cropped_img_to_show)
    # show()
    cv2.imshow('handWriting Capture Cropped Image', cropped_img_to_show)
    cv2.imshow('handWriting Capture Cropped Thresh', cropped_thresh_to_Show)

    # pause = True





SCALE_FACTOR = 4 if len(sys.argv) == 1 else int(sys.argv[1])
video_capture = cv2.VideoCapture(0)

while True:
    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break

    # Capture frame-by-frame
    ret, frame = video_capture.read()


    frame = cv2.resize(frame, (frame.shape[1]/SCALE_FACTOR,frame.shape[0]/SCALE_FACTOR))

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Digit Detection
    if key == ord('c'):
        print "Pressed"
        pause = not pause
        analyze_digit(frame)
        pause = not pause


    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    if not pause:
        cv2.imshow('handWriting Capture', frame)

    cv2.putText(frame, "Press ESC or 'q' to quit. \nPress 'c' to analyze picture", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()