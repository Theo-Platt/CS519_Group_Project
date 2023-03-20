# Code to train models to recognizer symbols and everything. 

import cv2
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from skimage.transform import rescale
from misc import HogTransformer
from sklearn.model_selection import train_test_split


# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

CSV_PATH= Path("./data/symbol_dataset.csv")

# nums
nums = ['0','1','2','3','4','5','6','7','8','9']
# characters
chars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#operators
operators= ['(', ')',\
            '+', '-', '×', '÷', '=']
comma = ","
   

PICTURE_WIDHT=100
PICTURE_HEIGHT=100
#https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
if __name__ == "__main__":
    nums_dataset = {}
    with open(CSV_PATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for row in reader:
            if row[1] in nums:
                if row[1] not in nums_dataset:
                    nums_dataset[row[1]] = []
                temp = nums_dataset[row[1]]
                path = Path(row[0])
                my_data = cv2.imread(str(path))
        
                
                my_data = cv2.cvtColor(my_data, cv2.COLOR_BGR2GRAY)
                temp.append(my_data)
    
    x = []
    y = []
    for num in nums_dataset:
        for my_data in nums_dataset[num]:
            x.append(my_data)
            y.append(num)
    # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    X_hog = hogify.fit_transform(x)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_hog)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)




    model = LogisticRegression(C=100, solver='lbfgs', max_iter=100, multi_class="ovr")
    model.fit(X_train, y_train)

    2
3
4
	
y_pred = model.predict(X_test)
print(np.array(y_pred == y_test)[:25])
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))
