# CS 519 M01
# Project Stage 3 - Piecewise Dataset Generator
# Authors: Theoderic Platt, Long Tran

#imports
from random import randint, Random
import random
from sympy import *
from shutil import which
import cv2
import numpy as np
import os
import imageio.v3 as iio
import csv
from settings.CONFIG import *
from misc import normalize_img
from train_codes.single_gen import save_latex



def generate_piecwise_dataset(height, instances_num):
    rand = Random()
    dataset=[]
    equationList = []
    #generate functions
    for j in range(instances_num):
        sample = random.sample(FULL_CLASSES,height)
        texStr = ""
        texStr += "\\["
        texStr += "   \\left\\{"
        texStr += "\\begin{array}{ll}"
        for i in sample:
            texStr += "      "+str(i)+"\\\\"
        texStr += "\\end{array} "
        texStr += "\\right. "
        texStr += "\\]"
        equationList.append(texStr)
    #generate png from functions
    counter = 0
    for equation in equationList: 
        try:
            counter+=1
            folder_path = PIECWISE_PATH / f'piecewise_{height}_folder'
            if not os.path.exists(str(folder_path)):
                os.makedirs(folder_path)

            print(f"Generating instance ({counter}/{instances_num}) for piecwise height: {height}                ", end='\r')
            path = folder_path / f'piecwise_{height}({counter}).png'
            save_latex(path, equation, font=rand.choice(FONTS), dvi_density=randint(DENSITY_MIN, DENSITY_MAX))

            dataset.append([f"{str(path)}", f"{str(height)}"])
        except:
            pass
    return dataset
    print("\nfinished generating piecwise")


def main():
    print("How many instances to generate?")
    size = int(input())
    dataset=[]

    try:
        dataset.extend(generate_piecwise_dataset(height=1, instances_num=size))
        dataset.extend(generate_piecwise_dataset(height=2, instances_num=size))
        dataset.extend(generate_piecwise_dataset(height=3, instances_num=size))
        dataset.extend(generate_piecwise_dataset(height=4, instances_num=size))
    except: 
        print("Something went wrong while generating the data.")    
    finally:
        save_file_name = "piecewise_dataset.csv"
        
        # save
        print(f"Saving to {save_file_name}...")
        with open(str(DATA_FOLDER / save_file_name), 'w') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerows(dataset)
                f.close()
