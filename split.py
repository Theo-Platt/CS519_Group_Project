# A program to try to split the images using histogram. 
# Code is based on https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
# and https://towardsdatascience.com/segmentation-in-ocr-10de176cf373
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt

# segmentize image
# not quite recursive right now because the recursion does not work. 
def segmentize_recursive(img):
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgs = segmentize_recur(add_padding(black_or_white_transformer(img)))
    return imgs

# 0 if space
# not 0 if not space. 
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
    for x in np.nditer(img):
        if x != 255:
            return False
    
    return True

def add_padding(img):
    row, col= get_shape(img)
    temp1 = np.array([[np.uint8(255) for i in range(col + 2)] for j in range(row + 2)])
    temp1[1:row+1, 1:col+1] = img
    return temp1

def get_shape(img):
    return img.shape

def segmentize(img):
     # the rgb of white is (255, 255, 255)
    gray_img = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        if space_found and value > 0 and i > 0:
            #cv2.line(src, (0, i-1), (img_col, i-1), (255, 0, 0))
            rows.append(i-1)
            space_found = False
        elif value == 0:
            if not space_found:
                #cv2.line(src, (0, i), (img_col, i), (255, 0, 0))
                rows.append(i)
            space_found = True
            
        
        prev = value

    space_found = False
    for i in range(len(vertical_hist[0])):
        value = vertical_hist[0][i]
        if space_found and value > 0 and i > 0:
            #cv2.line(src, (i-1, 0), (i-1, img_row), (255, 0, 0))
            cols.append(i-1)
            space_found = False
        elif value == 0:
            if not space_found:
                #cv2.line(src, (i, 0), (i, img_row), (255, 0, 0))
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
            
            temp =  gray_img[row_s:row_e, col_s:col_e]
            temp = add_padding(temp)
            sub_imgs.append(
               temp
            )

    # check for empty images. 
    result = []
    for img in sub_imgs:
        shape = get_shape(img)
        if shape[0] == 0 or shape[1] == 0:
            continue
        if img_empty(img):
            continue
            
        result.append(img)
    
    # Plot histogram
    # plt.plot(horizontal_hist, range(img_row))
    # plt.xlim(img_col)
    # plt.xlabel('aaaRows')
    # plt.ylabel('Frequency')
    # plt.show()

    return result

# segmentize recursive
def segmentize_recur(img, old_img=np.array([[]])):
    if get_shape(img) == get_shape(old_img):
        return [img]
    
    sub_imges = segmentize(img)
    result = []
    for sub_img in sub_imges:
        result.extend(segmentize_recur(sub_img, img))
        
        
    
    return result

# pre-process the pixels
def black_or_white_transformer(img):
    for i in range(len(img)):
        row = img[i]
        for j in range(len(row)):
            black_diff = np.abs(row[j] - 0)
            white_diff = np.abs(row[j] - 255)
            
            row[j] = 0
            if black_diff > white_diff:
                row[j] = np.uint8(255)

    return img
if __name__ == "__main__":
    src = cv2.imread("test.png")
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Source image', black_or_white_transformer(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    if src is None:
        print('Could not open or find the image.')
        exit(0)
    
    imgs = segmentize_recursive(src)
    print("sub images:", len(imgs))
    
    i = 0
    for img in imgs:
        cv2.imshow(f'cropped{i}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        if img_empty(img):
            print("empty")
        
    
        i = i + 1
  
