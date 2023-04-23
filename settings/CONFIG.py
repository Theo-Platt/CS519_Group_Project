# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

DATA_FOLDER = Path("./data/")
NUM_PATH = DATA_FOLDER / "nums"
CHAR_PATH = DATA_FOLDER / "chars" 
OP_PATH = DATA_FOLDER / "operators" 
PIECWISE_PATH = DATA_FOLDER / "piecwise"
SINGLE_GEN_CSV_PATH= DATA_FOLDER / "symbol_dataset.csv"
PIECEWISE_GEN_CSV_PATH= DATA_FOLDER / "piecewise_dataset.csv"
MODEL_FOLDER = Path("./trained_models/")

############
### Data ###
############
# nums
NUMS_CLASSES = ['0','1','2','3','4','5','6','7','8','9']
# characters
CHARS_CLASSES_LOWER= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
CHARS_CLASSES_UPPER = list(x.upper() for x in CHARS_CLASSES_LOWER)
CHARS_CLASSES = CHARS_CLASSES_LOWER
CHARS_CLASSES.extend(CHARS_CLASSES_UPPER) 
#operators
OPERATORS_CLASSES =['(', ')','+', '-', '=',',','divide','times','curly_bracket' ]
OPERATORS_DICT  ={
    'divide':'÷', 
    'times':'×',
    'curly_bracket':'\{'
}
# all
FULL_CLASSES = []
FULL_CLASSES.extend(NUMS_CLASSES)
FULL_CLASSES.extend(CHARS_CLASSES)
FULL_CLASSES.extend(OPERATORS_CLASSES)

# piecewise
PIECEWISE_CLASSES = ['1','2','3','4']


# picture settings
PICTURE_WIDHT=100
PICTURE_HEIGHT=100
DENSITY_MIN = 100
DENSITY_MAX = 600

# FONTs
# Link: https://tex.stackexchange.com/questions/33677/setting-font-family-for-the-whole-document
# https://www.overleaf.com/learn/latex/Font_typefaces#Changing_the_default_document_fonts
FONTS = [("mathptmx", "cmr"), ("lmodern","lmr"), ("helvet","phv"), ("tgtermes","qtm"), ("tgpagella","qpl"), ("mathptmx", "ptm"), ("tgschola", "qcs")]

# COLOR
WHITE = (255, 255, 255)

# saved model path
PIPELINE_PATH=Path("./model_parameters/pipe.sav")
NUM_MODEL_PATH=Path("./model_parameters/num.sav")
CHAR_MODEL_PATH=Path("./model_parameters/char.sav")
OP_MODEL_PATH=Path("./model_parameters/op.sav")

#############      
### Tests ###
#############
# testing values: tuple of (actual_value, correct_model)
#note: evaluate will convert the model classificaton as follows
#      1: "NUMBERS"
#      2: "OPERATORS"
#      3: "CHARACTERS"
#note: Test 1 does not work as the equals sign is read improperly by the image parser.
TEST_1 = ("2 x - 3 = - 7 1",                # actual symbols in image 
          [1,3,2,1,2,2,1,1],                # correct model to classify symbols by
          Path('./eval_codes/test1.png')) # path to test1.png image

TEST_2 = ("1 2 3 4 5 6 7 8 9 1 0 1 1 1 2 1 4", # actual symbols in image 
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # correct model to classify symbols by
          Path('./eval_codes/test2.png'))      # path to test2.png image

TEST_3 = ("3 + 4 - 5 X a 5 + a x B",      # actual symbols in image 
          [1,2,1,2,1,3,3,1,2,3,3,3],      # correct model to classify symbols by
          Path('./eval_codes/test3.png')) # path to test3.png image

TEST_4 = ("T H I S I S A T E S T 4 2 6 4 1 2 8 + -", # actual symbols in image 
          [3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,2,2], # correct model to classify symbols by
          Path('./eval_codes/test4.png'))            # path to test4.png image

TEST_INPUT = TEST_2
TEST_IMAGE = TEST_INPUT[2]
