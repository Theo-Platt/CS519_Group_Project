# import segmentation library
from func_codes.split import segmentize_recursive
# misc
from misc import normalize_img
import cv2

class Converter:
    def __init__(self):
        self.pipeline = None
        self.i = 0
    
    # convert an image to latex
    def convert_img_to_latex(self, img):
        # segmentize the image
        imgs = segmentize_recursive(img)
        latex_maps = self.convert(imgs)

    # convert an image map to LaTex
    def convert(self, imgs_bundle):
        if imgs_bundle is None:
            return
        
        # the source image
        src_img = imgs_bundle[0]
        # it sub images 
        imgs_map = imgs_bundle[1]

        # resize image
        src_img = normalize_img(src_img)
        
        cv2.imshow(f'src_image {self.i}', src_img)
        self.i += 1
        
        for i in range(len(imgs_map)):
            imgs_row = imgs_map[i]
            for j in range(len(imgs_row)):
                sub_img_bundle = imgs_row[j]
                self.convert(sub_img_bundle)

    def predict(self, img):
        pass