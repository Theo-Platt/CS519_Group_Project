from torch.nn import Sequential, Conv2d, Conv, ReLU, MaxPool2d

class DNNClassifierBase:
    def __init__(self):
        model = Sequential()
        model.add_module('conv1', Conv2d(in_channels=1, out_channels=32,kernel_size=5, padding=2))
        model.add_module('relu1', ReLU())
        model.add_module('pool1', MaxPool2d(kernel_size=2))
        model.add_module('conv2', Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=2))
        model.add_module('relu2', ReLU())
        model.add_module('pool2', MaxPool2d(kernel_size=2))
        self.model = model
    
    # fit the model to the data
    # inptut:
    # + X: A NumPy 2D matrix containing the features fo each instance. 
    # + y: A NumPy vector containing the class labels. 
    def fit(X, y):
        pass

    # transform the data. 
    # input: Expecting a NumPy 2D matrix containing the features of each instance. 
    def transform(X):
        pass