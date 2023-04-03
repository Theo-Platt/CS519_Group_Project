# import segmentation library
from func_codes.split import segmentize_recursive
# misc
from misc import normalize_img, move_center, load_models
import cv2
import numpy as np

class Converter:
    def __init__(self):
        self.pipeline = None
        self.i = 0
        self.models = load_models()
        self.model_classifier = load_models(classifier=True)
        
    
    # convert an image to latex
    def convert_img_to_latex(self, img):
        # segmentize the image
        imgs = segmentize_recursive(img)
        latex_maps = self.convert(imgs)
        return latex_maps

    # convert an image map to LaTex
    # not finished
    def convert(self, imgs_bundle):
        if imgs_bundle is None:
            return " "
        
        # the source image
        src_img = imgs_bundle[0]
        # it sub images 
        imgs_map = imgs_bundle[1]

        # resize image
        src_img = normalize_img(src_img)
        # predict it
        result = self.predict(src_img)

        for i in range(len(imgs_map)):
            imgs_row = imgs_map[i]
            for j in range(len(imgs_row)):
                sub_img_bundle = imgs_row[j]
                result += self.convert(sub_img_bundle)
            result += "\n"
        
        return result

    def predict(self, src_img):
        #print(img.shape)
        result = ""
    
        selected_model = self.model_classifier.predict(np.array([src_img]))
        result = self.models[selected_model[0]].predict(np.array([src_img]))

        return result[0]