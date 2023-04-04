import cv2
from pathlib import Path
from misc import move_center
from misc import load_models
import numpy as np
from func_codes.split import segmentize_recursive
from settings import CONFIG
from misc import normalize_img
from misc import run_test_input_accuracy


predicted_symbols = []
predicted_models = []
model_evaluator = {
    "NUMBERS": 1,
    "OPERATORS":2,
    "CHARACTERS":3,
}

def guess_recursive(imgs_bundle, models, model_classifier):
    if imgs_bundle is None:
        return
    
    src_img = imgs_bundle[0] 
    imgs_map = imgs_bundle[1]

    x = src_img.shape[0]
    y = src_img.shape[1]
    if len(imgs_map) == 1 and x <= 100 and y <= 100:
        src_img = move_center(src_img)
        src_img = normalize_img(src_img)

        # add predicted model to solution
        selected_model = model_classifier.predict(np.array([src_img]))
        pred  = models[selected_model[0]].predict(np.array([src_img]))

        selected = selected_model[0]
        selected = model_evaluator[selected]
        predicted_models.append(selected)
        predicted_symbols.extend(pred)

        # cv2.imshow(f'src_image', src_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
    
    for i in range(len(imgs_map)):
        imgs_row = imgs_map[i]
        for j in range(len(imgs_row)):
            sub_img_bundle = imgs_row[j]
            guess_recursive(sub_img_bundle, models, model_classifier)

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
    
    #prepare expected values
    expected_values = CONFIG.TEST_INPUT[0].replace(" ", "")
    expected_values = [*expected_values]

    #make predictions
    imgs = segmentize_recursive(src)
    guess_recursive(imgs, models,model_classifier)
    
    #accuracy of model prediction
    print("\nModel Selection accuracy")
    run_test_input_accuracy(CONFIG.TEST_INPUT[1],predicted_models)
    
    #accuracy of symbol prediction
    print("\nSymbol Prediction accuracy")
    run_test_input_accuracy(expected_values,predicted_symbols)

    #accuracy of symbol prediction filtering out misclassified models
    filtered_expected_values =[]
    filtered_predicted_symbols =[]
    for i in range(len(predicted_models)):
        if predicted_models[i] == CONFIG.TEST_INPUT[1][i]:
            filtered_expected_values.append(expected_values[i])
            filtered_predicted_symbols.append(predicted_symbols[i])
    print("\nSymbol Prediction accuracy (excluding improperly classified model selection)")
    run_test_input_accuracy(filtered_expected_values,filtered_predicted_symbols)

    