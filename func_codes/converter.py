# Codes to convert an image to its equivalent true content. 
# Date: Spring 2023, NMSU
# Author: Long, Theo

# import segmentation library
from func_codes.split import segmentize_recursive, segmentize_col_nocolor, segmentize_recursive_nocolor, segmentize_row
# misc
from misc import normalize_img, move_center, load_models, OPERATORS_DICT
import cv2
import numpy as np

# Class to convert from an image to its actual content. 
# Originally, we planned to convert into latex. However, we did not have enough time to look up how to write latex. 
class Converter:
    # Constructor
    def __init__(self):
        self.pipeline = None
        self.i = 0

        # load the trained machine learning
        self.models = load_models()
        self.model_classifier = load_models(specific="classifier")
        self.model_piecewise  = load_models(specific="piecewise")
    
    # convert an entire image to its actual content
    def convert_img_to_latex(self, img):
        row_imgs = segmentize_row(img)
        result = ""
        for rowimg in row_imgs: 
            if rowimg is None:
                continue
            # segmentize the image
            imgs = segmentize_recursive_nocolor(rowimg)
            # actual conversion
            result += self.convert(imgs)
            result += "\n"
        return result

    # convert an image bundle (gotten from the segmentize_recursive() function) to its actualy content.
    # top is to idnetify that we are at the top of the recursion
    def convert(self, imgs_bundle, top=True):
        # if there is not image bundle, return empty string
        if imgs_bundle is None:
            return ""
        
        # the source image
        src_img = imgs_bundle[0]
        # it sub images 
        imgs_map = imgs_bundle[1]

        #special function for leaf node
        # if the image has more white pixels than a certain number of pixels, don't predict it
        def majority_empty(img):
            sum = 0
            pixels = 0
            tmp = np.nditer(img)
            for x in tmp:
                pixels += 1
                if x == 255:
                   sum += 1

            # checking if the number of white pixels exceeded certain threshold
            if (sum / pixels) > 0.95:
                return True
            
            return False
            
        
        
        # leaf node
        # predict the image
        #only does this on leaf
        result = ""
        if len(imgs_map) <= 1:
            # checking if its majority is empty space
            # if we got into leaf nodes, then if an image has empty spaces, it is a mistake of the segmentation function due to that there is gray color instead of just white (255) and black (0). 
            if majority_empty(src_img):
                return ""  
            
            #resize image
            src_img = normalize_img(src_img)   
            if src_img is None:
                return ""  
            
            # cv2.imshow(f'src_image', src_img)
            # cv2.setWindowProperty('src_image', cv2.WINDOW_AUTOSIZE, 1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #   
            #predict it
            result = self.predict(src_img)
            #print("predicted to be", result)

            # if the prediction is times
            # return x 
            # if the prediction is curly bracket
            # returns {
            if result == "times" :
                return OPERATORS_DICT[result]
            if result == "curly_bracket":
                return '{'

            # return right away
            return result


        # checking each sub-image
        curly = None
        non_curly = None
        for i in range(len(imgs_map)):
            # get the row of sub-image
            imgs_row = imgs_map[i]
            empty_row = True
            # check each iamge
            for j in range(len(imgs_row)):
                sub_img_bundle = imgs_row[j]
                # if there is at least one image, then this row is not empty
                if sub_img_bundle is not None:
                    empty_row = False
                elif top: # if we are at top, then add a space
                    result += " "    
                
                # convert this sub-image bundle
                imgs_row[j] = self.convert(sub_img_bundle, top=False)

                # if it is curly brace, save that we see a curly bracket
                if imgs_row[j] == "{":
                    curly = j
                elif imgs_row[j] != "":
                    non_curly = j

                # add the img to the result                    
                result += imgs_row[j]

            # if we are at the end of each row
            # then, add a newline   
            if top and not empty_row:
                result += "\n"
        
        # checking if we are seeing a piecewise. 
        # if there is a curly bracket in this image, and there are also other stuffs. Then, it must be a piecewise. 
        if (not curly is None) and (not non_curly is None):
            # get the sub-image that is to the right of the curly bracket
            curly_img = segmentize_col_nocolor(src_img, curly)

            # if the image found is not None
            # predict it
            if not curly_img is None:
                # cv2.imshow(f'src_image', curly_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()  

                result = "curly (\n" 
                imgs = segmentize_recursive_nocolor(curly_img)
                result += self.convert(imgs)
                result += "\n)"

        # special case for equal.
        if len(imgs_map) == 5 and len(imgs_map[0]) == 3:
            if imgs_map[1][1] == "-" and imgs_map[3][1] == "-":
                # resize image
                src_img = normalize_img(src_img)

                # predict it to double check
                temp = self.predict(src_img)
                if temp == "=":
                    return temp

        # special case for division
        if len(imgs_map) == 7 and len(imgs_map[0]) == 3:
            if imgs_map[1][1] != None and imgs_map[3][1] == "-" and imgs_map[5][1] != None:
                # resize image
                src_img = normalize_img(src_img)

                # predict it for double check
                temp = self.predict(src_img)
                if temp == "divide":
                    return "÷"
                
        # special case for i
        if len(imgs_map) == 5 and len(imgs_map[0]) == 3:
                # resize image
                src_img = normalize_img(src_img)

                # predict to double check
                temp = self.predict(src_img)
                if temp == "i" or temp == "I":
                    return temp
             
        return result

    # predict 
    def predict(self, src_img):
        result = ""

        # predict its class, either character, operators, or numbers
        selected_model = self.model_classifier.predict(np.array([src_img]))
        # in each class, predict its correct symbol
        result = self.models[selected_model[0]].predict(np.array([src_img]))

        return result[0]