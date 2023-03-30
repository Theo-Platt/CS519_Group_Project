# Code to generate images of each individual characters, number digits, and operators

from random import randint
from sympy import *
from shutil import which
import cv2
import numpy as np
import os
import imageio.v3 as iio
import csv
from CONFIG import *
from misc import move_center

# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

def save_latex(path, latex, dvi_density):
    with open(path, 'wb') as outputfile:
            preview(latex, viewer='BytesIO', outputbuffer=outputfile, dvioptions=['-D',str(dvi_density)])
        
    img = cv2.imread(str(path))
    

    result = move_center(img)
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
    for num in NUMS_CLASSES:
        folder_path = NUM_PATH / f'num_{num}_folder'
        if not os.path.exists(str(folder_path)):
             os.makedirs(folder_path)

        for i in range(instances_num):    
            path = folder_path / f'num_{num}({i}).png'
            save_latex(path, num, randint(DENSITY_MIN, DENSITY_MAX))

            dataset.append([f"{str(path)}", f"{str(num)}"])
    print("finished generating num pictures")

    # # characters
    # for char in CHARS_CLASSES:
    #     folder_path = CHAR_PATH / f'char_{char}_folder'
    #     if not os.path.exists(str(folder_path)):
    #          os.makedirs(folder_path)

    #     for i in range(instances_num):    
    #         path = folder_path / f'char_{char}({i}).png'
    #         save_latex(path, char, randint(DENSITY_MIN, DENSITY_MAX))

    #         dataset.append([f"{str(path)}", f"{str(char)}"])

    # print("finished generating characters")

    # # operators
    # for op in OPERATORS_CLASSES:
    #     folder_path = OP_PATH / f'op_{op}_folder'
    #     if not os.path.exists(str(folder_path)):
    #         os.makedirs(folder_path)

    #     for i in range(instances_num):    
    #         path = folder_path / f'op_{op}({i}).png'
    #         save_latex(path, op, randint(DENSITY_MIN, DENSITY_MAX))

    #         dataset.append([f"{str(path)}", f"{str(op)}"])
    # print("finished generating operators")
    
    # # comma
    # for comma in COMMAS_CLASSES:
    #     folder_path = COMMA_PATH
    #     if not os.path.exists(str(folder_path)):
    #         os.makedirs(folder_path)

    #     for i in range(instances_num):    
    #         path = folder_path / f'comma({i}).png'
    #         save_latex(path, ",", randint(DENSITY_MIN, DENSITY_MAX))

    #         dataset.append([f"{str(path)}", f"comma"])
    # print("finished generating comma")
    
    save_file_name = "symbol_dataset.csv"
    
    # save
    print(f"Saving to {save_file_name}...")
    with open(str(DATA_FOLDER / save_file_name), 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(dataset)
            f.close()

