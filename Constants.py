""" Constants.py

This script contains constant values in a class with static variables
"""

import os


class Constants:
    """Contains all constant values to be used in the project"""

    TRAIN_DOC_DIR_PATH = os.path.join('data', 'train')
    TEST_DOC_DIR_PATH = os.path.join('data', 'test')
    TEST_DOC_NAME = 'Circular-Buisness-Models.pdf'
    # TEST_DOC_NAME = 'GradientDescentInMachineLearning.pdf'

    CATEGORIES = ["Business", "Cancer", "Chemistry", 'MachineLearning', 'Music']
    NUM_DOCS_PER_CATEGORY = 10
