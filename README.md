# CS519_Group_Project
CS-519-M01; Dr. Huiping Cao
## Authors 
- Long Tran <br/>
- Theoderic Platt <br/>


# Prerequisites:
- pip 23.0.1       -- only if dependencies are not yet installed.
- Python 3.10.10   -- all testing was conducted on this version of python.
- requirements.txt -- all dependencies

# Running the code
The codebase is run through two primary python scripts, both of which need additional parameters to be properly utilized.
## main
The main program runs our models against an inputted .png image containing a mathematical formula. The output of main is what our models identified as the text contained within the image file. To run main, do the following:
- 'python main -p <img>'
  - <img> should contain the path to the image file that you with to test. Some default images we have supplied for our testing are as follows:
    - 'python main -p input.png'
    - 'python main -p input2.png'

## tools
The tools section of the code is utilized to allow for the datasets to be generated, the models to be trained, or the models to be evaluated on premade test cases. 
### Generating data
Data can be generated by performing one of the following:<br/>
- 'python tools.py generate_symbols'
  - Provides options for generating characters data, numbers data, and operators data. Overwrites the local **./data/symbol_dataset.csv** with whatever data was run in this call. For proper usage, ensure that all three data sub-sets are used.
- 'python tools.py generate_piecewise'
  - Generates piecewise functions as data with heights 1, 2, 3, and 4. Number of instances for each height will be requested upon running.

### Training Models
For all model generation, if an existing model exists, you will be given the option to overwrite it with the newly generated model. If no such model exists, you will be given the option to save this model. Models will be saved as .bin files under the sub-directory **./trained_models**. Models can be trained by performing the following: <br/>
- 'python tools.py train'
  - Training single class models will train the characters, numbers, and operators models using Logistic Regression. 
  - Training the intraClass model will train a model containing all single class models, for distinguishing between which is needed. This is trained using a CNN, the architecture can be seen in the report.
  - Training the piecewise class model will train the piecewise dataset model using a CNN, the architecture can be seen in the report.

### Evaluating Models
Evaluating the models simply runs the model on a hardcoded pre-made test case and provides metrics for how accurately the intraClass model performs, how well the single class models perform, and how well these models perform together. To run model evaluation: <br/>
- 'python tools.py evaluate'

