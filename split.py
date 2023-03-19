# A program to try to split the data using histogram. 
# Code is based on https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
# and https://towardsdatascience.com/segmentation-in-ocr-10de176cf373
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentize(img):
     # the rgb of white is (255, 255, 255)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_row = len(gray_img)
    img_col =  len(gray_img[0])

    # If the img has foreground pixel as black i.e pixel value = 0
    horizontal_hist = img_col - np.sum(gray_img,axis=1,keepdims=True)/255
    # Logic :- No.of columns - No.of white pixels

    # If the img has foreground pixel as black i.e pixel value = 0
    vertical_hist = img_row - np.sum(gray_img,axis=0,keepdims=True)/255
    # Logic :- No.of columns - No.of white pixels

    # cut the images
    space_found = False
    for i in range(len(horizontal_hist)):
        value = horizontal_hist[i][0]

        if space_found and value > 0 and i > 0:
            cv2.line(src, (0, i-1), (img_col, i-1), (255, 0, 0))
            space_found = False
        elif value == 0:
            if not space_found:
                cv2.line(src, (0, i), (img_col, i), (255, 0, 0))
            space_found = True
        
        prev = value

    space_found = False
    for i in range(len(vertical_hist[0])):
        value = vertical_hist[0][i]
        if space_found and value != 0 and i > 0:
            cv2.line(src, (i-1, 0), (i-1, img_row), (255, 0, 0))
            space_found = False
        elif value == 0:
            if not space_found:
                cv2.line(src, (i, 0), (i, img_row), (255, 0, 0))
            space_found = True

     # Plot histogram
    # plt.plot(horizontal_hist, range(img_row))
    # plt.xlim(img_col)
    # plt.xlabel('Rows')
    # plt.ylabel('Frequency')
    # plt.show()

    

if __name__ == "__main__":
    src = cv2.imread("test.png")
  

    if src is None:
        print('Could not open or find the image.')
        exit(0)
   
    segmentize(src)
   
   
       
    cv2.imshow('Source image', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()