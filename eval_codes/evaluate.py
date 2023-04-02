import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score
from misc import move_center
from misc import load_models
import numpy as np
from func_codes.split import segmentize_recursive
from settings import CONFIG
from misc import normalize_img

expected_values = "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 1 - 2 + 4 + 6 + 7 + 9 + 1 + 1 - 1 1 × 2 × 3 - 2 + 1 + 2"
expected_values = "1 2 3 4 5 6 7 8 9 1 0 1 1 1 2 1 4"

predicted_numbers = []
predicted_ops = []

def guess_recursive(imgs_bundle, models):
    if imgs_bundle is None:
        return
    
    src_img = imgs_bundle[0] 
    imgs_map = imgs_bundle[1]

    x = src_img.shape[0]
    y = src_img.shape[1]
    if len(imgs_map) == 1 and x <= 100 and y <= 100:
        # x_ratio = 50/ src_img.shape[0]  
        # y_ratio = 50 / src_img.shape[1]
        # src_img = cv2.resize(src_img)
    
       
        src_img = move_center(src_img)
        src_img = normalize_img(src_img)
        # print("start")
       
        
        num  = models['NUMBERS'].predict(np.array([src_img]))
        #op = pipes['('].predict(np.array([src_img]))

        predicted_numbers.extend(num)
        #predicted_ops.extend(op)

        cv2.imshow(f'src_image', src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        # print("found leaf")
    
    
    for i in range(len(imgs_map)):
        imgs_row = imgs_map[i]
        for j in range(len(imgs_row)):
            sub_img_bundle = imgs_row[j]
            guess_recursive(sub_img_bundle, models)

def main(): 
    models = load_models()
    # print(models)
    # exit(0)
    try:
        src = cv2.imread(CONFIG.TEST_IMAGE.__str__())
        # cv2.imshow(f'src', src)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
    except:
        print('Could not open or find the image.')
        exit(0)
    else:
        if src is None:
            print('Could not open or find the image.')
            exit(0)
    

    imgs = segmentize_recursive(src)
    guess_recursive(imgs, models)
    
    expected_values = CONFIG.TEST2_VALUES.split(" ")
    print("Expected: ",expected_values)
    print("numbers:  ",predicted_numbers)
    print('Percentage correct: ', 100 *accuracy_score(expected_values, predicted_numbers), '%')