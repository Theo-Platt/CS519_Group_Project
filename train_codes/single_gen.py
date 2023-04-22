# Code to generate images of each individual characters, number digits, and operators

from random import randint, Random
from sympy import *
from shutil import which
import cv2
import numpy as np
import os
import imageio.v3 as iio
import csv
from settings.CONFIG import *
from misc import normalize_img
 
# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

def save_latex(path, latex, font, dvi_density):
    font_package, font_fam = font
    with open(path, 'wb') as outputfile:
            preamble = "\\documentclass[22pt]{minimal}\n"  + f"\\usepackage[T1]{{fontenc}} \\usepackage{{{font_package}}} \\renewcommand{{\\sfdefault}}{{{font_fam}}}\\renewcommand{{\\familydefault}}{{\sfdefault}}" + "\\begin{document}"
            preview(latex, viewer='BytesIO', outputbuffer=outputfile, preamble=preamble,dvioptions=['-D',str(dvi_density)])
        
    img = cv2.imread(str(path))
    img = normalize_img(img)

    # # view result
    # cv2.imshow("centered", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save result
    cv2.imwrite(str(path), img)



def main():

    do_nums=False
    do_chars=False
    do_ops=False
    if which('latex') is None:
        print("Latex has not been installed")
        exit(1)
    if which('dvipng') is None:
        print("dvipng has not been installed")
        exit(1)

    instances_num = int(input("Enter the number of pictures you want for each category: "))
    if str(input("Generate new NUMBERS data? (y/n)\n")) =='y': do_nums = True
    if str(input("Generate new CHARACTERS data? (y/n)\n")) =='y': do_chars = True
    if str(input("Generate new OPERATORS data? (y/n)\n")) =='y': do_ops = True

    if not do_nums and not do_chars and not do_ops and not do_comma:
        print("No data generation selected. Exiting.")
        exit(0)
    rand = Random()
    dataset = []
    try:
        if do_nums:
            # numbers
            for num in NUMS_CLASSES:
                folder_path = NUM_PATH / f'num_{num}_folder'
                if not os.path.exists(str(folder_path)):
                    os.makedirs(folder_path)
                    

                
                for i in range(instances_num):    
                    print(f"Generating instance {i}/{instances_num} for NUMBER {num}     ", end='\r')
                    path = folder_path / f'num_{num}({i}).png'
                    save_latex(path, num, font=rand.choice(FONTS), dvi_density=randint(DENSITY_MIN, DENSITY_MAX))

                    dataset.append([f"{str(path)}", f"{str(num)}"])
            print("finished generating num pictures           ")

        if do_chars:
            # characters
            
            for char in CHARS_CLASSES:
                lower=False
                if char.islower(): 
                    lower=True
                if lower: folder_path = CHAR_PATH / f'char_{char}_folder'
                else: folder_path = CHAR_PATH / f'char_{char}_Upper_folder'
                if not os.path.exists(str(folder_path)):
                    os.makedirs(folder_path)

                for i in range(instances_num):    
                    print(f"Generating instance {i}/{instances_num} for CHARACTER {char}     ", end='\r')
                    path = folder_path / f'char_{char}({i}).png'
                    save_latex(path, char, font=rand.choice(FONTS), dvi_density=randint(DENSITY_MIN, DENSITY_MAX))

                    dataset.append([f"{str(path)}", f"{str(char)}"])

            print("finished generating characters           ")

        if do_ops:
            # operators
            op_name = ''
            for op in OPERATORS_CLASSES:
                if op == '×': op_name = 'times'
                if op == '÷': op_name = 'divide'
                if op == '\{': op_name = 'curly_bracket'
                if op_name == '': folder_path = OP_PATH / f'op_{op}_folder'
                else: folder_path = OP_PATH / f'op_{op_name}_folder'
                if not os.path.exists(str(folder_path)):
                    os.makedirs(folder_path)

                for i in range(instances_num):    
                    print(f"Generating instance {i}/{instances_num} for OPERATOR {op}     ", end='\r')
                    if op_name == '': path = folder_path / f'op_{op}({i}).png'
                    else: path = folder_path / f'op_{op_name}({i}).png'
                    save_latex(path, op, font=rand.choice(FONTS), dvi_density=randint(DENSITY_MIN, DENSITY_MAX))

                    if op_name == '': dataset.append([f"{str(path)}", f"{str(op)}"])
                    else: dataset.append([f"{str(path)}", f"{str(op_name)}"])
            print("finished generating operators           ")

    finally:
        save_file_name = "symbol_dataset.csv"
        
        # save
        print(f"Saving to {save_file_name}...")
        with open(str(DATA_FOLDER / save_file_name), 'w') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerows(dataset)
                f.close()

