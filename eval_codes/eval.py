import cv2
from sklearn.metrics import accuracy_score
from misc import move_center
import numpy as np
from func_codes.split import segmentize_recursive
expected_values = "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 1 - 2 + 4 + 6 + 7 + 9 + 1 + 1 - 1 1 × 2 × 3 - 2 + 1 + 2"
expected_values = "1 2 3 4 5 6 7 8 9 1 0 1 1 1 2 1 4"

predicted_numbers = []
predicted_ops = []

def guess_recursive(imgs_bundle, pipes):
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

        # print("start")
       
        
        num  = pipes['0'].predict(np.array([src_img]))
        #op = pipes['('].predict(np.array([src_img]))

        predicted_numbers.extend(num)
        #predicted_ops.extend(op)

        cv2.imshow(f'src_image', src_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        # print("found leaf")
      
    
    for i in range(len(imgs_map)):
        imgs_row = imgs_map[i]
        for j in range(len(imgs_row)):
            sub_img_bundle = imgs_row[j]
            guess_recursive(sub_img_bundle, pipes)

if __name__ == "__main__":
    pipelines = []
    src = cv2.imread("test2.png")

    if src is None:
        print('Could not open or find the image.')
        exit(0)

    imgs = segmentize_recursive(src)
    guess_recursive(imgs, pipelines)
    
    expected_values = expected_values.split(" ")
    print("Expected: ",expected_values)
    print("numbers:  ",predicted_numbers)
    print('Percentage correct: ', accuracy_score(expected_values, predicted_numbers))