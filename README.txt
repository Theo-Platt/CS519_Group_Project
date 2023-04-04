# CS519_Group_Project
Long Tran & Theoderic T. Platt
CS-519-M01; Dr. Huiping Cao

#Prerequisites:
pip 23.0      -- only if dependencies are not yet installed.
Python 3.9.13 -- all testing was conducted on this version of python.

# Running the code
'pip install -r requirements.txt'
'python tools.py extension'

To run our project, an extension must be supplied. Available extensions are as follows:
generate -- generates a new dataset of the selected number of instances per class. WARNING: this will overwrite the existing dataset and can be very time-consuming.
train    -- Train new models based on the dataset. You will be queried on whether to overwrite or keep existing models.
evaluate -- Run the models through a test case in order to observe the behavior of the models and see accuracy metrics.

