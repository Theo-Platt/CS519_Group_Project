# Code to train models to recognizer symbols and everything. 

import numpy as np
from pathlib import Path
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
from settings import CONFIG
from os.path import join
import pickle
import time as tm

import cv2
from func_codes.split import segmentize_recursive, show_recursive

from train_codes.cnn_models import CNNClassifier1, CNNClassifierInter,CNNClassifierPiecewise


# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

from misc import parse_data

# create a pipeline that will go through the following
# hog transformer
# standard scaler
# PCA
# and finally the classifier model of choice. 
# The model should follow scikit-learn interface
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
    ]
    )

    return pipe

# save the pipeline 
def save_pipeline(pipe, class_name, X_test):
    # allow data model overwrite?
    try:
        # load old data
        old_model = pickle.load( open(join(MODEL_FOLDER,Path(f'{class_name}_MODEL.bin')), 'rb') )
        old_model.predict(X_test) #ensure old model exists
        # print('  Training percentage: ', 100 *accuracy_score(y_train, old_model.predict(X_train)),'%')
        # print('  Testing percentage:  ', 100 *accuracy_score(y_test, old_model.predict(X_test)),'%')
        save = input(f"Do you want to overwrite existing model of '{class_name}_MODEL'? (y/n)\n")
        while(True):
            if save == 'y':
                file = open(join(MODEL_FOLDER,Path(f'{class_name}_MODEL.bin')), 'wb+')
                pickle.dump(pipe,file)
                break
            if save == 'n':
                break
            save = input("please type one of the following. (y/n)\n")
    except:
        save = input(f"No existing model found. Would you like to save a new model for '{class_name}_MODEL'? (y/n)\n")
        while(True):
            if save == 'y':
                file = open(join(MODEL_FOLDER,Path(f'{class_name}_MODEL.bin')), 'wb+')
                pickle.dump(pipe,file)
                break
            if save == 'n':
                break
            save = input("please type one of the following. (y/n)\n")
            
#https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
def main():
    classes  = [(NUMS_CLASSES,'NUMBERS') , (CHARS_CLASSES,'CHARACTERS') , (OPERATORS_CLASSES,'OPERATORS')]
    pw_class =  (PIECEWISE_CLASSES,'PIECEWISE')
    labels = []
    X_intra = []
    y_intra = []
    X_pw=[]
    y_pw=[]
    # Which classes to train
    doSC=False #single classes (NUMBERS, CHARACTERS, OPERATORS)
    doIC=False #intra class
    doPW=False #piecewise class
    if input("Train single classes?  (y/n): ") == 'y': doSC=True
    if input("Train intraclass?      (y/n): ") == 'y': doIC=True
    if input("Train piecewise class? (y/n): ") == 'y': doPW=True
    

    for CLASSES in classes:
        dataset, X, y = parse_data(SINGLE_GEN_CSV_PATH, CLASSES[0])

        print(np.unique(y))
        labels.append(CLASSES[1])
        # add these data for the intra classes classifier
        for element in X:
            X_intra.append(element)
            y_intra.append(CLASSES[1])

        ####################
        ### single class ###
        ####################
        if doSC:
            # print(np.array(CLASSES[0]).shape)
            # print(CLASSES[0])
            print(f"Training dataset: {CLASSES[1]}")
            # print(y)
            # print(np.unique(y))

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

            # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
            # create the model
            model = LogisticRegression(C=10, solver='lbfgs', max_iter=10000, multi_class="ovr")
            # model = Perceptron()
            
            # train
            pipe = create_pipeline(model)
            # pipe = CNNClassifier1()
            start = tm.time()
            pipe.fit(X_train, y_train)
            end   = tm.time()
            training_time=(end-start)
            print(f"total time to train the model: {training_time} seconds")
            

            # test the data
            y_pred_train = pipe.predict(X_train)
            y_pred_test  = pipe.predict(X_test)
            print("This model has the following accuracy scores:")
            print('  Training percentage: ', 100 *accuracy_score(y_train, y_pred_train),'%')
            print('  Testing percentage:  ', 100 *accuracy_score(y_test, y_pred_test),'%')

            # save pipeline
            save_pipeline(pipe, CLASSES[1], X_test)


    ##################
    ### intraclass ###
    ##################
    if doIC:
        X = np.array(X_intra)
        y = np.array(y_intra)

        # print('\n\ny_intra: ',y_intra)

        print(f"Training intraclass:")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

        # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
        # create the model
        model = LogisticRegression(C=10, solver='lbfgs', max_iter=10000, multi_class="ovr")
        # model = Perceptron()
        print(np.unique(y))

        # train
        pipe = CNNClassifierInter(epochs=50, labels=labels)
        start = tm.time()
        pipe.fit(X_train, y_train)
        end   = tm.time()
        training_time=(end-start)
        print(f"total time to train the model: {training_time} seconds")

        # test the data
        y_pred_train = pipe.predict(X_train)
        y_pred_test  = pipe.predict(X_test)
        print("This model has the following accuracy scores:")
        print('  Training percentage: ', 100 *accuracy_score(y_train, y_pred_train),'%')
        print('  Testing percentage:  ', 100 *accuracy_score(y_test, y_pred_test),'%')

        # save pipeline
        save_pipeline(pipe, "CLASSIFIER", X_test)

    #################
    ### piecewise ###
    #################
    if doPW:
        dataset, X, y = parse_data(PIECEWISE_GEN_CSV_PATH, pw_class[0])

        labels.append(pw_class[1])
        # add these data for the intra classes classifier
        for element in X:
            X_pw.append(element)
            y_pw.append(CLASSES[1])

        X = np.array(X_pw)
        y = np.array(y_pw)
        # print('\n\ny_intra: ',y_intra)
        print(f"Training piecwise class:")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

        # train
        pipe = CNNClassifierPiecewise(epochs=20, labels=labels)
        start = tm.time()
        pipe.fit(X_train, y_train)
        end   = tm.time()
        training_time=(end-start)
        print(f"total time to train the model: {training_time} seconds")
            

        # test the data
        y_pred_train = pipe.predict(X_train)
        y_pred_test  = pipe.predict(X_test)
        print("This model has the following accuracy scores:")
        print('  Training percentage: ', 100 *accuracy_score(y_train, y_pred_train),'%')
        print('  Testing percentage:  ', 100 *accuracy_score(y_test, y_pred_test),'%')

        # save pipeline
        save_pipeline(pipe, "PIECEWISE", X_test)