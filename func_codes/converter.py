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
    def convert(self, imgs_bundle, top=True):
        if imgs_bundle is None:
            return ""
        
        # the source image
        src_img = imgs_bundle[0]
        # it sub images 
        imgs_map = imgs_bundle[1]

        # only does this on leaf
        result = ""
        if len(imgs_map) <= 1:
            # resize image
            src_img = normalize_img(src_img)

            #print(imgs_map.shape)
            cv2.imshow(f'src_image', src_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # predict it
            result = self.predict(src_img)

            return result
        

        # manage each individula subnodes. 
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
                result += self.convert(sub_img_bundle, top=False)
                
            if top and not empty_row:
                result += "\n"
        return result

    def predict(self, src_img):
        #print(img.shape)
        result = ""
    
        selected_model = self.model_classifier.predict(np.array([src_img]))
        result = self.models[selected_model[0]].predict(np.array([src_img]))

        return result[0]