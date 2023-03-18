# run the output through this: https://latex2image.joeraut.com/
# each line of output corresponds to one instance, what needs to be fixed is replacing
# the "print(texStr)" with a function call that does what the website is doing. The library
# here: https://pypi.org/project/pnglatex/ claims to do this, but I am having environment 
# path errors with my latex and cant get it working.
# I was also trying this most recently: https://stackoverflow.com/questions/1381741/converting-latex-code-to-images-or-other-displayble-format-with-python
# but ran into a similar path problem.

# change the num variable on line 14 and the size variable on line 15 to change output.

import random
from random import randint
from sympy import *
num=2    # number of items in the {
size=300 # number of instances to make

from sympy.utilities.misc import find_executable

from pnglatex import pnglatex

# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

# create all permutations of allowed symbols for { problems of size 2, 3, or 4.
def foo(arr,n_items):
    if n_items == 2:
        for a in arr:
            for b in arr:
                foo = random.sample(arr,n_items)
                # print("\\[")
                texStr = ""
                texStr += "   \\left\\{"
                texStr += "\\begin{array}{ll}"
                texStr += "      "+str(a)+"\\\\"
                texStr += "      "+str(b)+"\\\\"
                texStr += "\\end{array} "
                texStr += "\\right. "
                #print(texStr)

    if n_items == 3:
        pass
    for a in arr:
        for b in arr:
            for c in arr:
                foo = random.sample(arr,n_items)
                # print("\\[")
                texStr = ""
                texStr += "   \\left\\{"
                texStr += "\\begin{array}{ll}"
                texStr += "      "+str(a)+"\\\\"
                texStr += "      "+str(b)+"\\\\"
                texStr += "      "+str(c)+"\\\\"
                texStr += "\\end{array} "
                texStr += "\\right. "
                #print(texStr)

    for a in arr:
        for b in arr:
            for c in arr:
                for d in arr:
                    foo = random.sample(arr,n_items)
                    # print("\\[")
                    texStr = ""
                    texStr += "   \\left\\{"
                    texStr += "\\begin{array}{ll}"
                    texStr += "      "+str(a)+"\\\\"
                    texStr += "      "+str(b)+"\\\\"
                    texStr += "      "+str(c)+"\\\\"
                    texStr += "      "+str(d)+"\\\\"
                    texStr += "\\end{array} "
                    texStr += "\\right. "
                    #print(texStr)

    

# create n_items instances with random selection of characters
def bar(arr,n_items, sampleSize = 300):
    for j in range(sampleSize):
        foo = random.sample(arr,n_items)
        # print("\\[")
        texStr = ""
        texStr += "   \\left\\{"
        texStr += "\\begin{array}{ll}"
        for i in foo:
        #   if randint(1,10) > 1:
            texStr += "      "+str(i)+"\\\\"
            # else:
            #     texStr += "      \\frac{"+str(i)+"}{"+str(random.choice(arr))+"}\\\\"
        texStr += "\\end{array} "
        texStr += "\\right. "
        # print("\\]")

        # preview(texStr,viewer='file',filename='test.png')
        #print(texStr)
        return texStr

NUM_PATH = Path("./data/nums")
CHAR_PATH = Path("./data/chars")


if __name__ == "__main__":
    #print(find_executable('latex'))
    # numbers
    nums = ['0','1','2','3','4','5','6','7','8','9']
    for num in nums:
        path = NUM_PATH / f'num_{num}.png'
        with open(path, 'wb') as outputfile:
            preview(num, viewer='BytesIO', outputbuffer=outputfile)
    print("finished generating num pictures")

    # characters
    chars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    temp = list(x.upper() for x in chars)
    chars.extend(temp)
   
    for char in chars:
        path = CHAR_PATH / f'char_{char}.png'
        with open(path, 'wb') as outputfile:
            preview(char, viewer='BytesIO', outputbuffer=outputfile)
    print("finished generating characters")


    operators= ['(','+','-']
    
    total = []
    total.extend(nums)
    total.extend(chars)
    total.extend(operators)
    
    #equation = bar(total,num,size)

   


# run the output through this: https://latex2image.joeraut.com/
# each line of output corresponds to one instance, what needs to be fixed is replacing
# the "print(texStr)" with a function call that does what the website is doing. The library
# here: https://pypi.org/project/pnglatex/ claims to do this, but I am having environment 
# path errors with my latex and cant get it working.
