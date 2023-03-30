# Code to train models to recognizer symbols and everything. 

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler, Normalizer
from skimage.transform import rescale
from transformer import HogTransformer, SpreadTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from CONFIG import *
from sklearn.decomposition import PCA

import cv2
from split import segmentize_recursive, show_recursive


# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

from misc import parse_data
expected_values = "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 1 - 2 + 4 + 6 + 7 + 9 + 1 + 1 - 1 1 × 2 × 3 - 2 + 1 + 2"
expected_values = "1 2 3 4 5 6 7 8 9 1 0 1 1 1 2 1 4"

predicted_numbers = []
predicted_ops = []

def create_pipeline(model):
    pipe = Pipeline(
    [
        # ("hogify", HogTransformer(
        #     pixels_per_cell=(14, 14), 
        #     cells_per_block=(2,2), 
        #     orientations=9, 
        #     block_norm='L2-Hys')
        # ), 
        ("spreader", SpreadTransformer()),
        ('scaler', StandardScaler()),
        ("dim_reduct", PCA(n_components=0.9)),
        ('model', model)
    ])

    return pipe


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
    
        result = np.full((100, 100), fill_value=255, dtype=np.uint8)
        # set offsets for top left corner
        xx = 0
        yy = 0
        x = src_img.shape[0]
        y = src_img.shape[1]
        # print(result.shape)
        # print(src_img.shape)
        result[yy:yy+x, xx:xx+y] = src_img
        src_img = result

        print("start")
       
        
        num  = pipes['0'].predict(np.array([src_img]))
        #op = pipes['('].predict(np.array([src_img]))

        predicted_numbers.extend(num)
        #predicted_ops.extend(op)

        cv2.imshow(f'src_image', src_img)
        print("found leaf")
      
    
    for i in range(len(imgs_map)):
        imgs_row = imgs_map[i]
        for j in range(len(imgs_row)):
            sub_img_bundle = imgs_row[j]
            guess_recursive(sub_img_bundle, pipes)
            

#https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
if __name__ == "__main__":
    pipelines = {}
    classes = [NUMS_CLASSES]
    for CLASSES in classes:
        dataset, X, y = parse_data(SINGLE_GEN_CSV_PATH, CLASSES)

        print(np.unique(y))

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
        # create the model
        #model = LogisticRegression(C=10, solver='lbfgs', max_iter=10000, multi_class="ovr")
        model = Perceptron()

        # train
        pipe = create_pipeline(model)
        pipe.fit(X_train, y_train)

        # test the data
        y_pred = pipe.predict(X_test)
        print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

        pipelines[CLASSES[0]] = pipe


    src = cv2.imread("test2.png")

  
    if src is None:
        print('Could not open or find the image.')
        exit(0)

    imgs = segmentize_recursive(src)
    guess_recursive(imgs, pipelines)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    
    expected_values = expected_values.split(" ")
    print("Expected: ",expected_values)
    print("numbers:  ",predicted_numbers)
    # print("ops:      ",predicted_ops)
    print(predicted_numbers)
    print(len(expected_values))
    print(len(predicted_numbers))
    # print(len(predicted_ops))
    print('Percentage correct: ', accuracy_score(expected_values, predicted_numbers))
    # print('Percentage correct: ', accuracy_score(expected_values, predicted_ops))