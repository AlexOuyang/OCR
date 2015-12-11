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
        The corpped images are saved under OCR/pics/cropped

        Press 'q' or 'ESC' to quit

        Press '1' to increase resized video frame for analysis by 20px
        
        Press '2' to decrease resized video frame for analysis by 20px

Run:
        python main.py SCALE_FACTOR

SCALE_FACTOR controls the frame size: 
frame size = frame size / SCALE_FACTOR


"""
import sys
import cv2
import numpy as np
from pylab import imread, imshow, imsave, figure, show, subplot, plot, scatter, title
import ocr
import multilayerPerceptron as mlp

print(__doc__)


ANALYZE = False    # Used to analyse video frame for digits
SCALE_FACTOR = 2 if len(sys.argv) == 1 else int(sys.argv[1])

frame_new_dim = 100


# Build neural network on start
print ("Building Multilayer Perceptron Network From Trained Model\n")
mlp_classifier = mlp.build_classifier('../trainedResult/model.npz')


def analyze_digit_MLP(img):
    """ Takes in an image matrix, crops out the digits and outputs it to file """

    ocr.delete_files("../pics/cropped/")

    print ("Preprocessing Image, Cropping Digits Into 28 X 28 Image Matrices\n")
    cropped_img_to_show, cropped_thresh_to_Show, cropped_digits = ocr.save_digit_to_binary_img_as_mnist(img, dim = 28, saveToFile = True, imgSize = frame_new_dim)

    print ("Image Preprocessing Done, %d Potential Digits Were Cropped Out\n" % len(cropped_digits))

    print ("Predicting Results\n")
    print ("Image    Digit     probability")

    index = 0
    for input_digit in cropped_digits:
        path = "../pics/cropped/" + str(index) + ".png"
        input_digit = imread(path)
        digit, probability = mlp.predict(input_digit, mlp_classifier)
        print ("%d.png      %d         %f" % (index, digit, probability))
        index += 1

    new_dim = (SCALE_FACTOR * img.shape[1]/2, SCALE_FACTOR * img.shape[0]/2)
    cropped_img_to_show = cv2.resize(cropped_img_to_show, new_dim)
    cropped_thresh_to_Show = cv2.resize(cropped_thresh_to_Show, new_dim)
    cv2.imshow('handWriting Capture Cropped Image', cropped_img_to_show)
    cv2.imshow('handWriting Capture Cropped Thresh', cropped_thresh_to_Show)



def analyze_digit_SVM(img):
    ocr.delete_files("../pics/cropped/")

    print ("Preprocessing Image, Cropping Digits Into 28 X 28 Image Matrices\n")
    cropped_img_to_show, cropped_thresh_to_Show, cropped_digits = ocr.save_digit_to_binary_img_as_mnist(img, dim = 8, saveToFile = True, imgSize = frame_new_dim)

    print ("Image Preprocessing Done, %d Potential Digits Were Cropped Out\n" % len(cropped_digits))

    print ("Predicting Results\n")
    print ("Image    Digit     probability")

    # index = 0
    # for input_digit in cropped_digits:
    #     path = "../pics/cropped/" + str(index) + ".png"
    #     input_digit = imread(path)
    #     digit, probability = mlp.predict(input_digit, mlp_classifier)
    #     print ("%d.png      %d         %f" % (index, digit, probability))
    #     index += 1

    # new_dim = (SCALE_FACTOR * img.shape[1]/2, SCALE_FACTOR * img.shape[0]/2)
    # cropped_img_to_show = cv2.resize(cropped_img_to_show, new_dim)
    # cropped_thresh_to_Show = cv2.resize(cropped_thresh_to_Show, new_dim)
    # cv2.imshow('handWriting Capture Cropped Image', cropped_img_to_show)
    # cv2.imshow('handWriting Capture Cropped Thresh', cropped_thresh_to_Show)


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
    if key == ord('n'):
        print "Sent frame for analysis"
        ANALYZE = not ANALYZE
        analyze_digit_MLP(frame)
        ANALYZE = not ANALYZE

    if key == ord('s'):
        print "Sent frame for analysis"
        ANALYZE = not ANALYZE
        analyze_digit_SVM(frame)
        ANALYZE = not ANALYZE

    if key == ord('1'):
        if frame_new_dim <= 500: frame_new_dim += 20
        print "Resized the frame for analysis to be ", frame_new_dim, " px"

    if key == ord('2'):
        if frame_new_dim >= 20: frame_new_dim -= 20
        print "Resized the frame for analysis to be ", frame_new_dim, " px"
    

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    cv2.imshow('handWriting Capture', frame)

    cv2.putText(frame, "Press ESC or 'q' to quit. \nPress 'c' to analyze picture", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()