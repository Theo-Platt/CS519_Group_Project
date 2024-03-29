# A program to try to split the images using histogram. 
# Code is based on https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
# and https://towardsdatascience.com/segmentation-in-ocr-10de176cf373
# Date: Spring 2023, NMSU
# Author: Long, Theo

from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from misc import add_padding, black_or_white_transformer, get_shape, img_empty

# convert the image to a grayscale image, then
# segmentize image
def segmentize_recursive(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs = segmentize_recur(add_padding(img))
    return imgs

# assuming the input is a grayscale image
# segmentize image
def segmentize_recursive_nocolor(img):
    imgs = segmentize_recur(add_padding(img))
    return imgs

# get the horizontal histogram
# 0 if space
# not 0 if not space. 
def get_horizontal_hist(img, col):
    # If the img has foreground pixel as black i.e pixel value = 0
    horizontal_hist = col - np.sum(img,axis=1,keepdims=True)/255
    # Logic :- No.of columns - No.of white pixels
    return horizontal_hist

# get the vertical histogram
# 0 if space
# not 0 if not space
def get_vertical_hist(img, row):
    # If the img has foreground pixel as black i.e pixel value = 0
    vertical_hist = row - np.sum(img,axis=0,keepdims=True)/255
    # Logic :- No.of columns - No.of white pixels

    return vertical_hist

# segmentize 
# assumes the input is a grayscale color. 
def segmentize(img):
    # the rgb of white is (255, 255, 255)
    gray_img = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_row = len(gray_img)
    img_col =  len(gray_img[0])

    # vertical hist
    horizontal_hist = get_horizontal_hist(gray_img, img_col)

    # vertical_hsit
    vertical_hist = get_vertical_hist(gray_img, img_row)

    # get the list of horizontal spaces
    rows = []
    cols = []
    space_found = False
    for i in range(len(horizontal_hist)):
        value = round(horizontal_hist[i][0], 0)

        if space_found and value > 0 and i > 0:
            #cv2.line(src, (0, i-1), (img_col, i-1), (255, 0, 0))
            rows.append(i-1)
            space_found = False
        elif value <= 0:
            if not space_found:
                #cv2.line(src, (0, i), (img_col, i), (255, 0, 0))
                rows.append(i)
            space_found = True
            
        
        prev = value
    
    # get the list of vertical spaces
    space_found = False
    for i in range(len(vertical_hist[0])):
        value = round(vertical_hist[0][i],)
        if space_found and value > 0 and i > 0:
            #cv2.line(src, (i-1, 0), (i-1, img_row), (255, 0, 0))
            cols.append(i-1)
            space_found = False
        elif value <= 0:
            if not space_found:
                #cv2.line(src, (i, 0), (i, img_row), (255, 0, 0))
                cols.append(i)
            space_found = True

    # crop the images. 
    # check for empty
    # add padding
    # add each sub-image to the sub-image matrix
    final_result = np.array([[None for i in range(len(cols))] for j in range(len(rows))])
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            row_s = rows[i]
            row_e = rows[i+1]
            col_s = cols[j]
            col_e = cols[j+1]
            
            # crop
            img =  gray_img[row_s:row_e, col_s:col_e]

            # empty
            shape = get_shape(img)
            if shape[0] == 0 or shape[1] == 0:
                continue
            if img_empty(img):
                continue

            # add padding
            img = add_padding(img)

            # add to final result
            final_result[i][j] = img

    # Plot histogram
    # plt.plot(horizontal_hist, range(img_row))
    # plt.xlim(img_col)
    # plt.xlabel('aaaRows')
    # plt.ylabel('Frequency')
    # plt.show()

    return final_result

# segmentize recursive
# segmentize an image and then segmentize each of its sub-image
# the input is a grayscale image
def segmentize_recur(img, old_img=np.array([[]])):
    # base case
    # if the sub-image has the same shape as the old one
    # this only happens if the image has nothing more to be segmented
    if get_shape(img) == get_shape(old_img):
        return (img, np.array([[]]))
    
    # segmentize this image
    sub_imges_map = segmentize(img)

    # recursively segmentize its sub-image
    row, col= sub_imges_map.shape
    for i in range(row):
        for j in range(col):
            sub_img = sub_imges_map[i][j]
            if sub_img is None:
                continue
            sub_img_map = segmentize_recur(sub_img, img)
            sub_imges_map[i][j] = sub_img_map
    
    return (img, sub_imges_map)

# get the image to the right of a symbol
# index is the index of the leftmost space of that symbol
# for example, if there is: space1 { space2 ...... Then, index should the 2nd index of the 2nd space in space2. See segmentation for more details.
# it's because the segmentation is done by seeing either a space when not seeing space, or when seeing a space, and then a non-space, then a space must be in the previous position.
def segmentize_col_nocolor(img, index):
    # the rgb of white is (255, 255, 255)
    gray_img = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_row = len(gray_img)
    img_col =  len(gray_img[0])

    # vertical_hsit
    vertical_hist = get_vertical_hist(gray_img, img_row)

    # get the boundary
    cols = []

    # get the list of vertical spaces
    space_found = False
    for i in range(len(vertical_hist[0])):
        value = round(vertical_hist[0][i],)
        if space_found and value > 0 and i > 0:
            #cv2.line(src, (i-1, 0), (i-1, img_row), (255, 0, 0))
            cols.append(i-1)
            space_found = False
        elif value <= 0:
            if not space_found:
                #cv2.line(src, (i, 0), (i, img_row), (255, 0, 0))
                cols.append(i)
            space_found = True

    # crop the images. 
    # check for empty
    # padding
    # add add the result
    final_result = np.array([None for i in range(len(cols))])

    row_s = 0
    row_e = img_row
    col_s = cols[index+1]
    col_e = img_col
    
    # crop
    img =  gray_img[row_s:row_e, col_s:col_e]

    # empty
    shape = get_shape(img)
    if shape[0] == 0 or shape[1] == 0:
        return None
    if img_empty(img):
        return None

    # add padding
    img = add_padding(img)

    return img

# segmentize by row
# not used
def segmentize_row(img):
    img = add_padding(img)
    gray_img = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_row = len(gray_img)
    img_col =  len(gray_img[0])

    
    # vertical hist
    horizontal_hist = get_horizontal_hist(gray_img, img_col)

    # get the boundary
    rows = []
    space_found = False
    for i in range(len(horizontal_hist)):
        value = round(horizontal_hist[i][0], 0)

        if space_found and value > 0 and i > 0:
            #cv2.line(src, (0, i-1), (img_col, i-1), (255, 0, 0))
            rows.append(i-1)
            space_found = False
        elif value <= 0:
            if not space_found:
                #cv2.line(src, (0, i), (img_col, i), (255, 0, 0))
                rows.append(i)
            space_found = True
            
        
        prev = value

    # crop the images. 
    # check for empty
    # padding
    # add add the result
    final_result = np.array([ None for j in range(len(rows))])
    for i in range(len(rows)-1):
        row_s = rows[i]
        row_e = rows[i+1]
        col_s = 0
        col_e = img_col
        
        # crop
        img =  gray_img[row_s:row_e, col_s:col_e]

        # cv2.imshow(f'src_image', img)
        # cv2.setWindowProperty('src_image', cv2.WINDOW_AUTOSIZE, 1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  

        # empty
        shape = get_shape(img)
        if shape[0] == 0 or shape[1] == 0:
            continue
        if img_empty(img):
            continue

        # add padding
        img = add_padding(img)

        # add to final result
        final_result[i] = img

    return final_result

# show the image recursively
# not used
def show_recursive(imgs_bundle):
    if imgs_bundle is None:
        return
    src_img = imgs_bundle[0]
    imgs_map = imgs_bundle[1]
    cv2.imshow(f'src_image', src_img)

    for i in range(len(imgs_map)):
        imgs_row = imgs_map[i]
        for j in range(len(imgs_row)):
            sub_img_bundle = imgs_row[j]
            if sub_img_bundle is None:
                continue

            sub_img = sub_img_bundle[0]
            cv2.imshow(f'cropped ({i}, {j})', sub_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    for i in range(len(imgs_map)):
        imgs_row = imgs_map[i]
        for j in range(len(imgs_row)):
            sub_img_bundle = imgs_row[j]
            show_recursive(sub_img_bundle)
            
# test program
if __name__ == "__main__":
    src = cv2.imread("test1.png")

  
    if src is None:
        print('Could not open or find the image.')
        exit(0)
    
    imgs = segmentize_recursive(src)
    show_recursive(imgs)


  
