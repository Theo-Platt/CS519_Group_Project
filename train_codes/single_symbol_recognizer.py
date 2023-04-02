# Code to train models to recognizer symbols and everything. 

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler, Normalizer
from skimage.transform import rescale
from train_codes.transformer import HogTransformer, SpreadTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from settings.CONFIG import *
from sklearn.decomposition import PCA
from misc import move_center
from sklearn.tree import DecisionTreeClassifier

import cv2
from func_codes.split import segmentize_recursive, show_recursive


# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

from misc import parse_data

def create_pipeline(model):
    pipe = Pipeline(
    [
        ("hogify", HogTransformer(
            pixels_per_cell=(14, 14), 
            cells_per_block=(2,2), 
            orientations=9, 
            block_norm='L2-Hys')
        ), 
        # ("spreader", SpreadTransformer()),
        ('scaler', StandardScaler()),
        ("dim_reduct", PCA(n_components=0.9)),
        ('model', model)
    ])

    return pipe
            
#https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
def main():
    classes = [NUMS_CLASSES]
    for CLASSES in classes:
        dataset, X, y = parse_data(SINGLE_GEN_CSV_PATH, CLASSES)

        print(np.unique(y))

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
        # create the model
        # model = LogisticRegression(C=10, solver='lbfgs', max_iter=10000, multi_class="ovr")
        # model = Perceptron()
        model= DecisionTreeClassifier()

        # train
        pipe = create_pipeline(model)
        pipe.fit(X_train, y_train)

        # test the data
        y_pred = pipe.predict(X_test)
        print('Percentage correct: ', accuracy_score(y_test, y_pred))