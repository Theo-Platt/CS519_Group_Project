# A program to try to split the data using histogram. 
# Code is based on https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
# and https://towardsdatascience.com/segmentation-in-ocr-10de176cf373
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt

# EMPTY THRESHOLD
# the sum of either horizontal or vertical that is greater than this means that it is empty
EMPTY_THRESHOLD = 10

def get_horizontal_hist(img, col):
    # If the img has foreground pixel as black i.e pixel value = 0
    horizontal_hist = col - np.sum(img,axis=1,keepdims=True)/255
    # Logic :- No.of columns - No.of white pixels
    return horizontal_hist


def get_vertical_hist(img, row):
    # If the img has foreground pixel as black i.e pixel value = 0
    vertical_hist = row - np.sum(img,axis=0,keepdims=True)/255
    # Logic :- No.of columns - No.of white pixels

    return vertical_hist

def img_empty(img):
    row, col = get_shape(img)

    # vertical hist
    horizontal_hist = get_horizontal_hist(img, col)
    
    # sum all the values in the matrix
    result = np.sum(horizontal_hist)
    if result > EMPTY_THRESHOLD:
        return True

    return False

def get_shape(img):
    return img.shape

def segmentize(img):
     # the rgb of white is (255, 255, 255)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_row = len(gray_img)
    img_col =  len(gray_img[0])

    
    # vertical hist
    horizontal_hist = get_horizontal_hist(gray_img, img_col)

    # vertical_hsit
    vertical_hist = get_vertical_hist(gray_img, img_row)

    # get the boundary
    rows = []
    cols = []
    space_found = False
    for i in range(len(horizontal_hist)):
        value = horizontal_hist[i][0]

        if space_found and value > EMPTY_THRESHOLD and i > 0:
            cv2.line(src, (0, i-1), (img_col, i-1), (255, 0, 0))
            rows.append(i-1)
            space_found = False
        elif value == 0:
            if not space_found:
                cv2.line(src, (0, i), (img_col, i), (255, 0, 0))
                rows.append(i)
            space_found = True
            
        
        prev = value

    space_found = False
    for i in range(len(vertical_hist[0])):
        value = vertical_hist[0][i]
        if space_found and value > EMPTY_THRESHOLD and i > 0:
            cv2.line(src, (i-1, 0), (i-1, img_row), (255, 0, 0))
            cols.append(i-1)
            space_found = False
        elif value == 0:
            if not space_found:
                cv2.line(src, (i, 0), (i, img_row), (255, 0, 0))
                cols.append(i)
            space_found = True

    # crop the images. 
    sub_imgs = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            row_s = rows[i]
            row_e = rows[i+1]
            col_s = cols[j]
            col_e = cols[j+1]
            sub_imgs.append(
                gray_img[row_s:row_e, col_s:col_e]
            )

    # check for empty images. 
    result = []
    for img in sub_imgs:
        shape = get_shape(img)
        if shape[0] == 0 or shape[1] == 0:
            continue
        if not img_empty(img):
            continue
            
        result.append(img)
    
    # Plot histogram
    # plt.plot(horizontal_hist, range(img_row))
    # plt.xlim(img_col)
    # plt.xlabel('Rows')
    # plt.ylabel('Frequency')
    # plt.show()

    return result

    

if __name__ == "__main__":
    src = cv2.imread("test.png")
  

    if src is None:
        print('Could not open or find the image.')
        exit(0)
   
    imgs = segmentize(src)
    print("sub images:", len(imgs))
    
    cv2.imshow('Source image', src)
    
    i = 0
    for img in imgs:
        print(img.shape)
        cv2.imshow(f'cropped{i}', img)
        i = i + 1
       
    cv2.waitKey(0)
    cv2.destroyAllWindows()