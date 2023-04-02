# use this library to generate path that will work in both windows and linux
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

DATA_FOLDER = Path("./data/")
NUM_PATH = DATA_FOLDER / "nums"
CHAR_PATH = DATA_FOLDER / "chars" 
OP_PATH = DATA_FOLDER / "operators" 
COMMA_PATH = DATA_FOLDER / "comma"
SINGLE_GEN_CSV_PATH= DATA_FOLDER / "symbol_dataset.csv"

# nums
NUMS_CLASSES = ['0','1','2','3','4','5','6','7','8','9']
# characters
CHARS_CLASSES_LOWER= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
CHARS_CLASSES_UPPER = list(x.upper() for x in CHARS_CLASSES_LOWER)
CHARS_CLASSES = CHARS_CLASSES_LOWER
CHARS_CLASSES.extend(CHARS_CLASSES_UPPER) 

#operators
OPERATORS_CLASSES= ['(', ')','+', '-', '×', '÷', '=']
# comma
COMMAS_CLASSES = ['comma', "not comma"]

# picture settings
PICTURE_WIDHT=100
PICTURE_HEIGHT=100
DENSITY_MIN = 100
DENSITY_MAX = 600

# FONTs
fonts = ["mathptmx"]

# COLOR
WHITE = (255, 255, 255)

# saved model path
PIPELINE_PATH=Path("./model_parameters/pipe.sav")
NUM_MODEL_PATH=Path("./model_parameters/num.sav")
CHAR_MODEL_PATH=Path("./model_parameters/char.sav")
OP_MODEL_PATH=Path("./model_parameters/op.sav")

