# Code to generate images of each individual characters, number digits, and operators

from random import randint
from sympy import *
from shutil import which

# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

NUM_PATH = Path("./data/nums")
CHAR_PATH = Path("./data/chars")
OP_PATH = Path("./data/operators")

# nums
nums = ['0','1','2','3','4','5','6','7','8','9']
# characters
chars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#operators
operators= ['(', ')',\
            '+', '-', '×', '÷', '=']
   


if __name__ == "__main__":
    if which('latex') is None:
        print("Latex has not been installed")
        exit(1)
    if which('dvipng') is None:
        print("dvipng has not been installed")
        exit(1)

    # numbers
    for num in nums:
        path = NUM_PATH / f'num_{num}.png'
        with open(path, 'wb') as outputfile:
            preview(num, viewer='BytesIO', outputbuffer=outputfile)
    print("finished generating num pictures")

    # characters
    temp = list(x.upper() for x in chars)
    chars.extend(temp)
    for char in chars:
        path = CHAR_PATH / f'char_{char}.png'
        with open(path, 'wb') as outputfile:
            preview(char, viewer='BytesIO', outputbuffer=outputfile)
    print("finished generating characters")

    # operators
    for op in operators:
        path = OP_PATH / f'op_{op}.png'
        with open(path, 'wb') as outputfile:
            preview(op, viewer='BytesIO', outputbuffer=outputfile)
    print("finished generating operators")
 
    #equation = bar(total,num,size)


