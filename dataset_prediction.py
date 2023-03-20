import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Piecewise_classifier:
    classifier = None
    data = type('data',(),{})
    random_state = 0

    def __init__(self,random_state = 42):
        self.classifier = LogisticRegression()
        self.data = type('data',(),{})
        self.random_state = random_state

    def train(self, test_size = -1):
        try:
            X = pd.read_csv('./data/dataset/dataset/dataset.csv', header=None)      
            X_tmp = []
            i=0
            for tmp in X[0]:
                f=np.array(np.matrix(tmp))
                X_tmp.append(f[0]) 
            X = np.array(X_tmp)
            # print(X)
            # print(X.shape)
            # exit(0)
            y = pd.read_csv('./data/dataset/dataset/keys.csv', header=None)
            y=y.values.ravel()
            # print(X.shape,y.shape)
            if test_size > 0:
                self.data.X_train, self.data.X_test, self.data.y_train, self.data.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
            else: 
                self.data.X_train = X
                self.data.y_train = y
                self.data.X_test  = None
                self.data.y_test  = None
            self.classifier.fit(self.data.X_train,self.data.y_train)
        except IOError:
            print("One or more data files could not be detected. Please ensure that the data has properly been generated by piecewise_generator.py before running.")
            exit(1)
        

    def predict(self, X):
        prediction = self.classifier.predict(X)
        return prediction

    def internal_test(self):
        if self.data.y_test is None:
            print("internal test requires that the dataset was trained using a testing split.")
            exit(1)
        train_prediction = self.classifier.predict(self.data.X_train)
        test_prediction  = self.classifier.predict(self.data.X_test)
        training_accuracy   = accuracy_score(self.data.y_train, train_prediction)
        prediction_accuracy = accuracy_score(self.data.y_test , test_prediction)
        return (training_accuracy,prediction_accuracy)



if __name__ == "__main__":
    foo = Piecewise_classifier()
    foo.train(test_size = 0.2)
    bar = foo.internal_test()
    training_accuracy = '%.3f'%(bar[0]*100)
    testing_accuracy = '%.3f'%(bar[1]*100)
    print("training accuracy: "+training_accuracy+"%")
    print("testing accuracy:  "+testing_accuracy+"%")
