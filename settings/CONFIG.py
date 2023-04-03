# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

DATA_FOLDER = Path("./data/")
NUM_PATH = DATA_FOLDER / "nums"
CHAR_PATH = DATA_FOLDER / "chars" 
OP_PATH = DATA_FOLDER / "operators" 
COMMA_PATH = DATA_FOLDER / "comma"
SINGLE_GEN_CSV_PATH= DATA_FOLDER / "symbol_dataset.csv"
MODEL_FOLDER = Path("./trained_models/")
TEST_IMAGE = Path('./eval_codes/test3.png')

# nums
NUMS_CLASSES = ['0','1','2','3','4','5','6','7','8','9']
# characters
CHARS_CLASSES_LOWER= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
CHARS_CLASSES_UPPER = list(x.upper() for x in CHARS_CLASSES_LOWER)
CHARS_CLASSES = CHARS_CLASSES_LOWER
CHARS_CLASSES.extend(CHARS_CLASSES_UPPER) 

#operators
OPERATORS_CLASSES =['(', ')','+', '-', '=',',','×', '÷']
OPERATORS_CLASSES_COMPLEX=[]


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

# testing values
TEST2_VALUES = "1 2 3 4 5 6 7 8 9 1 0 1 1 1 2 1 4"
TEST3_VALUES = "3 + 4 - 5 = a 5 + a = B"

