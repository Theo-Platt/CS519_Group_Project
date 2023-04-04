# CS 519 M01
# Project Stage 3 - Piecewise Dataset Generator
# Authors: Theoderic Platt, Long Tran
# Purpose: Generates the desired number of png images of piecewise 
#          functions using random characters, digits, or operators. 
#          The data is then formatted to 100x100 pixels in size and 
#          stored as pixel data in a dataset.csv found in the 
#          /data/dataset/dataset directory. A keys.csv is generated 
#          in the same directory, such that the first instance of 
#          dataset.csv is of the class of the first instance in 
#          keys.csv and so on.


# imports
import imageio.v3 as iio
import numpy      as np
import random
import csv
import cv2
import os
from sympy.utilities.misc import find_executable
from pnglatex             import pnglatex
from random               import randint
from pathlib              import Path             # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from sys                  import argv
from sympy                import *

# environment variables
# NUM_PATH = Path("./data/nums")
# CHAR_PATH = Path("./data/chars")
DATA_PATH = [
    Path("./data/dataset/02"),
    Path("./data/dataset/03"),
    Path("./data/dataset/04"),
    Path("./data/dataset/dataset")
    ]
enableChars = False
enableNums = False
disableP2 = False
disableP3 = False
disableP4 = False
debug = False

# # create all permutations of allowed symbols for { problems of size 2, 3, or 4.
# def foo(arr,n_items):
#     if n_items == 2:
#         for a in arr:
#             for b in arr:
#                 foo = random.sample(arr,n_items)
#                 # print("\\[")
#                 texStr = ""
#                 texStr += "   \\left\\{"
#                 texStr += "\\begin{array}{ll}"
#                 texStr += "      "+str(a)+"\\\\"
#                 texStr += "      "+str(b)+"\\\\"
#                 texStr += "\\end{array} "
#                 texStr += "\\right. "
#                 #print(texStr)

#     if n_items == 3:
#         pass
#     for a in arr:
#         for b in arr:
#             for c in arr:
#                 foo = random.sample(arr,n_items)
#                 # print("\\[")
#                 texStr = ""
#                 texStr += "   \\left\\{"
#                 texStr += "\\begin{array}{ll}"
#                 texStr += "      "+str(a)+"\\\\"
#                 texStr += "      "+str(b)+"\\\\"
#                 texStr += "      "+str(c)+"\\\\"
#                 texStr += "\\end{array} "
#                 texStr += "\\right. "
#                 #print(texStr)

#     for a in arr:
#         for b in arr:
#             for c in arr:
#                 for d in arr:
#                     foo = random.sample(arr,n_items)
#                     # print("\\[")
#                     texStr = ""
#                     texStr += "   \\left\\{"
#                     texStr += "\\begin{array}{ll}"
#                     texStr += "      "+str(a)+"\\\\"
#                     texStr += "      "+str(b)+"\\\\"
#                     texStr += "      "+str(c)+"\\\\"
#                     texStr += "      "+str(d)+"\\\\"
#                     texStr += "\\end{array} "
#                     texStr += "\\right. "
#                     #print(texStr)

# resize images to 100x100 pixels with generated image in upper left corner
def resize_all(size=-1, data=-1):
    if data==-1 or size==-1:
        print("improper use of resize_all function. data parameter must be supplied.")
        exit(1)
    #Source: https://stackoverflow.com/questions/61379067/adding-two-different-sized-images-or-padding-white-pixels-to-make-it-to-bigger-s
    for i in range(size):
        # read image
        img = cv2.imread('./data/dataset/0'+str(data)+'/P'+str(data)+'_'+str(i)+'.png')
        ht, wd, cc= img.shape
        # create new image of desired size and color (white) for padding
        ww = 100 
        hh = 100
        color = (255,255,255)
        result = np.full((hh,ww,cc), color, dtype=np.uint8)
        # set offsets for top left corner
        xx = 0
        yy = 0
        # copy img image into center of result image
        result[yy:yy+ht, xx:xx+wd] = img
        # save result
        cv2.imwrite('./data/dataset/0'+str(data)+'/P'+str(data)+'_'+str(i)+'.png', result)

# create n_items instances with random selection of characters
def bar(arr,n_items, sampleSize = 300):
    if n_items < 2 or n_items >4:
        print("program only supports 2, 3, or 4 tall functions.")
        print("n_iter = "+str(n_items))
        exit(1)
    equationList = []
    #generate functions
    for j in range(sampleSize):
        foo = random.sample(arr,n_items)
        texStr = ""
        texStr += "\\["
        texStr += "   \\left\\{"
        texStr += "\\begin{array}{ll}"
        for i in foo:
            texStr += "      "+str(i)+"\\\\"
        texStr += "\\end{array} "
        texStr += "\\right. "
        texStr += "\\]"
        equationList.append(texStr)
    #generate png from functions
    counter = 0
    for equation in equationList:
        path = DATA_PATH[n_items-2] / f'P{n_items}_{counter}.png'
        counter+=1
        with open(path, 'wb') as outputfile:
            preview(str(equation), viewer='BytesIO', outputbuffer=outputfile)
    # resize all images to 100 x 100 pixel
    resize_all(data = n_items, size=sampleSize)
    print("finished generating piecewise functions height: "+str(n_items))



if __name__ == "__main__":
    # handle CLI parameters
    args = set(argv)
    # if "-chars" in args: enableChars = True
    # if "-nums"  in args: enableNums  = True
    if "-p2"    in args: disableP2    = True
    if "-p3"    in args: disableP3    = True
    if "-p4"    in args: disableP4    = True

    print("WARNING: This program will overwrite everything inside the data directory.")
    while True:
        print("(y/n)")
        f=input()
        if str(f)=='y':
            break
        if str(f)=='n':
            exit(0)


    print("How many instances to generate?")
    size=input()
    size = int(size)

    if not os.path.exists('./data/dataset/dataset'):
        os.makedirs('./data/dataset/dataset')
    if not os.path.exists('./data/dataset/02'):
        os.makedirs('./data/dataset/02')
    if not os.path.exists('./data/dataset/03'):
        os.makedirs('./data/dataset/03')
    if not os.path.exists('./data/dataset/04'):
        os.makedirs('./data/dataset/04')


    #setup characters, numbers, and operators
    chars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    temp = list(x.upper() for x in chars)
    chars.extend(temp)
    nums = ['0','1','2','3','4','5','6','7','8','9']
    operators= ['(','+','-']
    total = []
    total.extend(nums)
    total.extend(chars)
    total.extend(operators)

    # # numbers
    # if enableNums:
    #     for num in nums:
    #         path = NUM_PATH / f'num_{num}.png'
    #         with open(path, 'wb') as outputfile:
    #             preview(num, viewer='BytesIO', outputbuffer=outputfile)
    #     print("finished generating num pictures")

    # # characters
    # if enableChars:
    #     for char in chars:
    #         path = CHAR_PATH / f'char_{char}.png'
    #         with open(path, 'wb') as outputfile:
    #             preview(char, viewer='BytesIO', outputbuffer=outputfile)
    #     print("finished generating characters")
    
    #create piecewise data
    if not disableP2: bar(total,2,size)
    if not disableP3: bar(total,3,size)
    if not disableP4: bar(total,4,size)

    # create actual dataset (only if all data was generated, b/c im lazy.)
    if not disableP2 and not disableP3 and not disableP4:
        # generate classes vector (order is important, so 3 loops are needed)
        classes=[]
        for i in range(size): classes.append(2)
        for i in range(size): classes.append(3)
        for i in range(size): classes.append(4)
        # print(classes)

        # generate dataset matrix
        # source: https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
        d2=[]
        d3=[]
        d4=[]
        dataset = []
        for i in range(size):
            im2 = iio.imread('./data/dataset/02/P2_'+str(i)+'.png')
            im3 = iio.imread('./data/dataset/03/P3_'+str(i)+'.png')
            im4 = iio.imread('./data/dataset/04/P4_'+str(i)+'.png')
            d2.append(im2)
            d3.append(im3)
            d4.append(im4)
        dataset.extend(d2)
        dataset.extend(d3)
        dataset.extend(d4)

        # write data to csv
        # source: https://www.geeksforgeeks.org/python-save-list-to-csv/
        with open('./data/dataset/dataset/dataset.csv', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(dataset)
            f.close()
        f = open("./data/dataset/dataset/keys.csv", 'w')
        for key in classes:
            f.write(str(key)+'\n')
        print("finished.")


    





