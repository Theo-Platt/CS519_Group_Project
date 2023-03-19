# Code to generate images of each individual characters, number digits, and operators

from random import randint
from sympy import *
from shutil import which
import cv2
import numpy as np
import os
import imageio.v3 as iio
import csv


# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

NUM_PATH = Path("./data/nums")
CHAR_PATH = Path("./data/chars")
OP_PATH = Path("./data/operators")
COMMA_PATH = Path("./data/comma")

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
DENSITY_MIN = 70
DENSITY_MAX = 600

def save_latex(path, latex, dvi_density):
    with open(path, 'wb') as outputfile:
            preview(latex, viewer='BytesIO', outputbuffer=outputfile, dvioptions=['-D',str(dvi_density)])
        
    img = cv2.imread(str(path))
    ht, wd, cc= img.shape

    # create new image of desired size and color (white) for padding
    ww = PICTURE_WIDHT
    hh = PICTURE_HEIGHT
    color = (255,255,255)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # set offsets for top left corner
    xx = 0
    yy = 0

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img

    # view result
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save result
    cv2.imwrite(str(path), result)


if __name__ == "__main__":
    if which('latex') is None:
        print("Latex has not been installed")
        exit(1)
    if which('dvipng') is None:
        print("dvipng has not been installed")
        exit(1)

    instances_num = int(input("Enter the number of pictures you want for each category: "))

    dataset = []
    # numbers
    for num in nums:
        folder_path = NUM_PATH / f'num_{num}_folder'
        if not os.path.exists(str(folder_path)):
             os.makedirs(folder_path)

        for i in range(instances_num):    
            path = folder_path / f'num_{num}({i}).png'
            save_latex(path, num, randint(DENSITY_MIN, DENSITY_MAX))

            dataset.append([f"{str(path)}", f"{str(num)}"])
    print("finished generating num pictures")

    # characters
    temp = list(x.upper() for x in chars)
    chars.extend(temp)
    for char in chars:
        folder_path = CHAR_PATH / f'char_{char}_folder'
        if not os.path.exists(str(folder_path)):
             os.makedirs(folder_path)

        for i in range(instances_num):    
            path = folder_path / f'char_{char}({i}).png'
            save_latex(path, char, randint(DENSITY_MIN, DENSITY_MAX))

            dataset.append([f"{str(path)}", f"{str(char)}"])

    print("finished generating characters")

    # operators
    for op in operators:
        folder_path = OP_PATH / f'op_{op}_folder'
        if not os.path.exists(str(folder_path)):
            os.makedirs(folder_path)

        for i in range(instances_num):    
            path = folder_path / f'op_{op}({i}).png'
            save_latex(path, op, randint(DENSITY_MIN, DENSITY_MAX))

            dataset.append([f"{str(path)}", f"{str(op)}"])
    print("finished generating operators")
    
    # comma
    for i in range(1):
        folder_path = COMMA_PATH
        if not os.path.exists(str(folder_path)):
            os.makedirs(folder_path)

        for i in range(instances_num):    
            path = folder_path / f'comma({i}).png'
            save_latex(path, comma, randint(DENSITY_MIN, DENSITY_MAX))

            dataset.append([f"{str(path)}", f"comma"])
    print("finished generating comma")
    
    save_file_name = "symbol_dataset.csv"
    
    # save
    print(f"Saving to {save_file_name}...")
    with open(f'./data/{save_file_name}', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(dataset)
            f.close()

