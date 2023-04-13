from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Dropout, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torch

import numpy as np

class CNNClassifier1:
    def __init__(self, epochs=100, img_size=(100, 100), batch_size=4):
        model = Sequential()
        model.add_module('conv1', Conv2d(in_channels=1, out_channels=32,kernel_size=5, padding=2))
        model.add_module('relu1', ReLU())
        model.add_module('pool1', MaxPool2d(kernel_size=2))
        model.add_module('conv2', Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=2))
        model.add_module('relu2', ReLU())
        model.add_module('pool2', MaxPool2d(kernel_size=2))
   
        model.add_module('flatten', Flatten())
        x = torch.ones((batch_size, 1, img_size[0], img_size[1]))
        size = model(x).shape

        model.add_module('fc1', Linear(size[1], 1024))
        model.add_module('relu3', ReLU())
        model.add_module('dropout', Dropout(p=0.5))
        model.add_module('fc2', Linear(1024, 10))

        self.model = model
        self.num_epochs = epochs
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=0.001)
        self.labels = None
        self.batch_size = 4
        self.img_size = img_size
    
    # fit the model to the data
    # inptut:
    # + X: A NumPy 2D matrix containing the features fo each instance. 
    # + y: A NumPy vector containing the class labels. 
    def fit(self, X, y):

        if self.labels is None:
            uniques = np.unique(y)
            self.labels = {}
            for i in range(len(uniques)):
                self.labels[uniques[i]] = i 

        # convert the type to the corresponding string. 
        new_y = []
        for i in range(len(y)):
            new_y.append(np.float32(self.labels[y[i]]))
        y = np.array(new_y)
        
    
        X = np.ones((800, 1, 100, 100))
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X = torch.from_numpy(X)
        y = torch.tensor(y)

        print(torch.ones((1, 28, 28)).shape)
        print("shape is", X.shape)
        print("shape is", y.shape)

        train_ds = TensorDataset(X, y)
        

        train_dl = DataLoader(train_ds, batch_size=4)
        #val_dl = DataLoader(val_ds, batch_size=128)

        loss_hist_train = [0] * self.num_epochs
        accuracy_hist_train = [0] * self.num_epochs
        loss_hist_valid = [0] * self.num_epochs
        accuracy_hist_valid = [0] * self.num_epochs
        for epoch in range(self.num_epochs):
            self.model.train()
            for x_batch, y_batch in train_dl:
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                #calculate loss, accuracy

    # predict the input
    # input: Expecting a NumPy 2D matrix containing the features of each instance. 
    def predict(self, X):
        X = torch.from_numpy(X)
        pred = self.model(X)
        print("prediction", pred)
        return pred