""" Main.py

This script is to be run the model on the dataset
"""

from Model import train_and_run_model

output = train_and_run_model()
print('Label for test document: ', output)
