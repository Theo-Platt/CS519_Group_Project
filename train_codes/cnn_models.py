from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Dropout, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

import numpy as np

class CNNClassifier1:
    def __init__(self, epochs=2, img_size=(100, 100), batch_size=64):
        model = Sequential()
        model.add_module('conv1', Conv2d(in_channels=1, out_channels=32,kernel_size=3, padding=2))
        model.add_module('relu1', ReLU())
        model.add_module('pool1', MaxPool2d(kernel_size=1))
        model.add_module('conv2', Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=2))
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
        self.batch_size = batch_size
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
        return pred
    
class CNNClassifierInter:
    def __init__(self, epochs=2, img_size=(100, 100), batch_size=64, labels=None):
        self.labels = {}
        self.nums = {}
        for i in range(len(labels)):
            self.labels[labels[i]] = i
            self.nums[i] = labels[i] 

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
        model.add_module('fc2', Linear(1024, len(self.labels)))

        self.model = model
        self.num_epochs = epochs
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=0.001)
        self.batch_size = batch_size


        self.img_size = img_size
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set the device
        self.model.to(self.device) # move the model to the right device
    
    # fit the model to the data
    # inptut:
    # + X: A NumPy 2D matrix containing the features fo each instance. 
    # + y: A NumPy vector containing the class labels. 
    def fit(self, X, y):
        # convert the type to the corresponding string. 
        new_y = []
        for i in range(len(y)):
            new_y.append(np.float32(self.labels[y[i]]))
        y = np.array(new_y)
        # convert type to float 32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        # convert x and y to tensor
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        # make a dataset
        train_ds = TensorDataset(X, y)
        # make a dataloader
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # training
        for epoch in range(self.num_epochs):
            print(f'Training epoch {epoch}/{self.num_epochs}',end='\r')
            self.model.train()
            for x_batch, y_batch in train_dl:
                # IMPORTANT
                # resize the batch from (batch size, 100, 100) to (batch size, 1, 100,100)
                num_batches = x_batch.shape[0] 
                x_batch = x_batch[:num_batches].view(num_batches, 1, 100, 100)
                # change the type of y
                y_batch = y_batch.type(torch.LongTensor)

                # move to the device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # actual trainnig
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                #calculate loss, accuracy
        print()

    # predict the input
    # input: Expecting a NumPy 2D matrix containing the features of each instance. 
    def predict(self, X):
        # convert numpy to tensor
        X = torch.from_numpy(X)
        # change the type
        X = X.type(torch.FloatTensor)
        
        # genearte a dataset
        pic_nums = X.shape[0] 
        dummy_y = torch.ones((pic_nums))
        pred_ds = TensorDataset(X, dummy_y)
        # generate dataloader
        pred_dl = DataLoader(pred_ds, batch_size=self.batch_size)

        # get the prediction
        final_answer = []
        for x_batch, y_patch in pred_dl:
            # change the batch size just like training
            num_batches = x_batch.shape[0] 
            x_batch = x_batch[:num_batches].view(num_batches, 1, 100, 100)
            x_batch = x_batch.to(self.device)

            # predictions
            preds = self.model(x_batch)
            preds = torch.argmax(preds, dim=1)
            final_answer.extend(preds.cpu().tolist())

        # map from num to class name
        final_answer = map(lambda x: self.nums[x], final_answer)
        final_answer = list(final_answer)
        return final_answer
    

    # test the accuracy using a testing dataset
    def test_accuracy(self, test_ds):
        test_dl = DataLoader(test_ds, batch_size=4)
        accuracy = 0

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device) 
                
                pred = self.predict(x_batch)
                is_correct = (pred == y_batch).float() 
                accuracy += is_correct.sum().cpu()
        accuracy /= len(test_dl.dataset)
        return 


class CNNClassifierPiecewise:
    def __init__(self, epochs=2, img_size=(100, 100), batch_size=64, labels=None):
        self.labels = {}
        self.nums = {}
        for i in range(len(labels)):
            self.labels[labels[i]] = i
            self.nums[i] = labels[i] 

        model = Sequential()
        model.add_module('conv1', Conv2d(in_channels=1, out_channels=32,kernel_size=5, padding=4))
        model.add_module('relu1', ReLU())
        model.add_module('pool1', MaxPool2d(kernel_size=2))
        model.add_module('conv2', Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=4))
        model.add_module('relu2', ReLU())
        model.add_module('pool2', MaxPool2d(kernel_size=2))
        model.add_module('conv3', Conv2d(in_channels=64, out_channels=128,kernel_size=5, padding=4))
        model.add_module('relu3', ReLU())
        model.add_module('pool3', MaxPool2d(kernel_size=2))
        

        model.add_module('flatten', Flatten())
        x = torch.ones((batch_size, 1, img_size[0], img_size[1]))
        size = model(x).shape

        model.add_module('fc1', Linear(size[1], 1024))
        model.add_module('relu3', ReLU())
        model.add_module('dropout', Dropout(p=0.5))
        model.add_module('fc2', Linear(1024, len(self.labels)))

        self.model = model
        self.num_epochs = epochs
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=0.001)
        self.batch_size = batch_size


        self.img_size = img_size
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set the device
        self.model.to(self.device) # move the model to the right device
    
    # fit the model to the data
    # inptut:
    # + X: A NumPy 2D matrix containing the features fo each instance. 
    # + y: A NumPy vector containing the class labels. 
    def fit(self, X, y):
        # convert the type to the corresponding string. 
        new_y = []
        for i in range(len(y)):
            new_y.append(np.float32(self.labels[y[i]]))
        y = np.array(new_y)
        # convert type to float 32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        # convert x and y to tensor
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        # make a dataset
        train_ds = TensorDataset(X, y)
        # make a dataloader
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # training
        for epoch in range(self.num_epochs):
            print(f'Training epoch {epoch}/{self.num_epochs}',end='\r')
            self.model.train()
            for x_batch, y_batch in train_dl:
                # IMPORTANT
                # resize the batch from (batch size, 100, 100) to (batch size, 1, 100,100)
                num_batches = x_batch.shape[0] 
                x_batch = x_batch[:num_batches].view(num_batches, 1, 100, 100)
                # change the type of y
                y_batch = y_batch.type(torch.LongTensor)

                # move to the device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # actual trainnig
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                #calculate loss, accuracy
        print()

    # predict the input
    # input: Expecting a NumPy 2D matrix containing the features of each instance. 
    def predict(self, X):
        # convert numpy to tensor
        X = torch.from_numpy(X)
        # change the type
        X = X.type(torch.FloatTensor)
        
        # genearte a dataset
        pic_nums = X.shape[0] 
        dummy_y = torch.ones((pic_nums))
        pred_ds = TensorDataset(X, dummy_y)
        # generate dataloader
        pred_dl = DataLoader(pred_ds, batch_size=self.batch_size)

        # get the prediction
        final_answer = []
        for x_batch, y_patch in pred_dl:
            # change the batch size just like training
            num_batches = x_batch.shape[0] 
            x_batch = x_batch[:num_batches].view(num_batches, 1, 100, 100)
            x_batch = x_batch.to(self.device)

            # predictions
            preds = self.model(x_batch)
            preds = torch.argmax(preds, dim=1)
            final_answer.extend(preds.cpu().tolist())

        # map from num to class name
        final_answer = map(lambda x: self.nums[x], final_answer)
        final_answer = list(final_answer)
        return final_answer
    

    # test the accuracy using a testing dataset
    def test_accuracy(self, test_ds):
        test_dl = DataLoader(test_ds, batch_size=4)
        accuracy = 0

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device) 
                
                pred = self.predict(x_batch)
                is_correct = (pred == y_batch).float() 
                accuracy += is_correct.sum().cpu()
        accuracy /= len(test_dl.dataset)
        return 