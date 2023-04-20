import cv2
from pathlib import Path
from misc import move_center
from misc import load_models
import numpy as np
from func_codes.split import segmentize_recursive
from settings import CONFIG
from misc import normalize_img
from misc import run_test_input_accuracy
from eval_codes.evaluate import guess_recursive


predicted_symbols = []
predicted_models = []
model_evaluator = {
    "NUMBERS": 1,
    "OPERATORS":2,
    "CHARACTERS":3,
}



def main(): 
    #load models and classifiers
    models = load_models()
    model_classifier = load_models(classifier=True)

    #read images
    try:
        src = cv2.imread(CONFIG.TEST_IMAGE.__str__())
        # cv2.imshow(f'src', src)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
    except:
        print('Could not open or find the image.')
        exit(0)
    else:
        if src is None:
            print('Could not open or find the image.')
            exit(0)

    #make predictions
    imgs = segmentize_recursive(src)
    predicted_models, predicted_symbols = guess_recursive(imgs, models,model_classifier)
    for sym in predicted_symbols:
        print(f'{sym}', end=' ')
    print('\n')
    