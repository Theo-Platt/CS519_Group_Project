import csv
import cv2
import numpy as np
from settings.CONFIG import *
from settings import CONFIG
import pickle
from os.path import join
from pathlib import Path

# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

def parse_data(path, labels):
    dataset = {}
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for row in reader:
            if row == []: continue
            # print(row)
            label = row[1]
            path = row[0]
            if label in labels:
                if label not in dataset:
                    dataset[label] = []
                
                # get the list of data associated with this label
                data_list = dataset[label]
                # get the data path
                path = Path(path)
                
                # read the image
                my_data = cv2.imread(str(path))
                # convert it to gray
                my_data = cv2.cvtColor(my_data, cv2.COLOR_BGR2GRAY)
                # black or white
                #my_data = black_or_white_transformer(my_data)
                # add it to the list
                data_list.append(my_data)
    
    X = []
    y = []
    for label in dataset:
        for my_data in dataset[label]:
            X.append(my_data)
            y.append(label)
    
    return (dataset, np.array(X), np.array(y))

def move_center_gen(img):
    ht, wd, cc= img.shape

    # create new image of desired size and color (white) for padding
    ww = PICTURE_WIDHT
    hh = PICTURE_HEIGHT
    color = WHITE
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # set offsets for top left corner
    xx = 0
    yy = 0
    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img
    return result


def move_center(img):
    ht, wd= img.shape

    # create new image of desired size and color (white) for padding
    ww = PICTURE_WIDHT
    hh = PICTURE_HEIGHT
    color = WHITE[0]
    result = np.full((hh,ww), color, dtype=np.uint8)

    # set offsets for top left corner
    xx = 0
    yy = 0
    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img
    return result

#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def resize(img):
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    return img

def normalize_img(img):
    trimmed_img = remove_whitespace(img)
    # if trimmed_img.shape[0] == 0 or trimmed_img.shape[1] == 0:
    #     cv2.imshow("centered", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    img = resize(trimmed_img)  
    return img

#https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
def remove_whitespace(img):
    gray = img
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 195).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    img = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    return img

# pre-process the pixels
def black_or_white_transformer(img):
    for i in range(len(img)):
        row = img[i]
        for j in range(len(row)):
            black_diff = np.abs(row[j] - 0)
            white_diff = np.abs(row[j] - 255)
            
            row[j] = 0
            if black_diff > white_diff:
                row[j] = np.uint8(255)

    return img

def img_empty(img):
    for x in np.nditer(img):
        if x != 255:
            return False
    
    return True

def get_shape(img):
    return img.shape

def add_padding(img):
    row, col= get_shape(img)
    temp1 = np.array([[np.uint8(255) for i in range(col + 2)] for j in range(row + 2)])
    temp1[1:row+1, 1:col+1] = img
    return temp1

def load_models(exitOnFail = False, verbose=True):
    # load models into dictionary
    models = {}
    keys = [
        'NUMBERS' , 
        'CHARACTERS' , 
        'OPERATORS'
        ]
    for cl in keys:
        try:
            f = open( join(CONFIG.MODEL_FOLDER,Path(f'{cl}_MODEL.bin')), 'rb')
            models[cl] = pickle.load(f)
            f.close()
            if verbose: print(f"successfully loaded model '{cl}_MODEL.bin'")
        except:
            if verbose: print(f"failed to load model '{cl}_MODEL.bin'")
            if exitOnFail: exit(1)
    return models