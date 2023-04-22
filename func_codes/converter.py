# import segmentation library
from func_codes.split import segmentize_recursive, segmentize_col_nocolor, segmentize_recursive_nocolor
# misc
from misc import normalize_img, move_center, load_models, OPERATORS_DICT
import cv2
import numpy as np

class Converter:
    def __init__(self):
        self.pipeline = None
        self.i = 0
        self.models = load_models()
        self.model_classifier = load_models(specific="classifier")
        self.model_piecewise  = load_models(specific="piecewise")
    
    # convert an image to latex
    def convert_img_to_latex(self, img):

        # segmentize the image
        imgs = segmentize_recursive(img)
        latex_maps = self.convert(imgs)
        return latex_maps

    # convert an image map to LaTex
    # not finished
    def convert(self, imgs_bundle, top=True):
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

            if (sum / pixels) > 0.95:
                return True
            
            return False
            

        #only does this on leaf
        result = ""
        if len(imgs_map) <= 1:
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
            #predict it
            
            result = self.predict(src_img)
            #print("predicted to be", result)
            if result == "times" :
                return OPERATORS_DICT[result]
            if result == "curly_bracket":
                return '{'

            return result

        curly = None
        non_curly = None
        for i in range(len(imgs_map)):
            imgs_row = imgs_map[i]
            empty_row = True
            for j in range(len(imgs_row)):
                sub_img_bundle = imgs_row[j]
                if sub_img_bundle is not None:
                    # show the image
                    empty_row = False
                elif top:
                    result += " "    
                imgs_row[j] = self.convert(sub_img_bundle, top=False)
                if imgs_row[j] == "{":
                    curly = j
                elif imgs_row[j] != "":
                    non_curly = j
                result += imgs_row[j]
                
            if top and not empty_row:
                result += "\n"
        
        
        if (not curly is None) and (not non_curly is None):
            curly_img = segmentize_col_nocolor(src_img, curly)

            if not curly_img is None:
                # cv2.imshow(f'src_image', curly_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()  

                result = "curly (\n" 
                imgs = segmentize_recursive_nocolor(curly_img)
                result += self.convert(imgs)
                result += "\n)"
        # needs updates
        # special case for equal.
        if len(imgs_map) == 5 and len(imgs_map[0]) == 3:
            if imgs_map[1][1] == "-" and imgs_map[3][1] == "-":
                 # resize image
                src_img = normalize_img(src_img)
                temp = self.predict(src_img)
                if temp == "=":
                    return temp

        # special case for division
        if len(imgs_map) == 7 and len(imgs_map[0]) == 3:
            #print("tu", imgs_map[1][1], imgs_map[3][1], imgs_map[5][1])
            if imgs_map[1][1] != None and imgs_map[3][1] == "-" and imgs_map[5][1] != None:
                 # resize image
                src_img = normalize_img(src_img)
                temp = self.predict(src_img)
                
                if temp == "divide":
                    return "÷"
                
        # special case for i
        if len(imgs_map) == 5 and len(imgs_map[0]) == 3:
                 # resize image
                src_img = normalize_img(src_img)
                temp = self.predict(src_img)
                if temp == "i" or temp == "I":
                    return temp

        # spacial 
             
        return result

    def predict(self, src_img):
        #print(img.shape)
        result = ""
    
        selected_model = self.model_classifier.predict(np.array([src_img]))
        result = self.models[selected_model[0]].predict(np.array([src_img]))

        return result[0]