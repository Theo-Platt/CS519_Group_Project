# import segmentation library
from func_codes.split import segmentize_recursive
# misc
from misc import move_center
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
        x = src_img.shape[0] # height
        y = src_img.shape[1] # width
        if x > 100 or y > 100:
            x_ratio = 50/ src_img.shape[0]  
            y_ratio = 50 / src_img.shape[1]
            src_img = cv2.resize(src_img, (100, 100))
        
        src_img = move_center(src_img)

        cv2.imshow(f'src_image {self.i}', src_img)
        self.i += 1
        
        for i in range(len(imgs_map)):
            imgs_row = imgs_map[i]
            for j in range(len(imgs_row)):
                sub_img_bundle = imgs_row[j]
                self.convert(sub_img_bundle)

    def predict(self, img):
        pass