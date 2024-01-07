""" DataPreprocessingUtils.py

This scripts contains code to preprocess the data to be used in Model.py
"""

import os
import nltk
import PyPDF2

from Constants import Constants


def read_pdf(pdf_path):
    """Read a single PDF"""

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


def process_text(text):
    """Remove punctuation and stopwords"""

    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]

    return ' '.join(tokens)


def read_all_train_pdfs():
    """Read all the PDFs in the train data"""

    category_texts = {}

    print('Parsing files:')
    for category in Constants.CATEGORIES:
        print("Category: ", category)
        category_path = os.path.join(Constants.TRAIN_DOC_DIR_PATH, category)

        if os.path.isdir(category_path):
            category_texts[category] = []

            for pdf_file in os.listdir(category_path):
                pdf_path = os.path.join(category_path, pdf_file)

                if os.path.isfile(pdf_path) and pdf_file.endswith('.pdf'):
                    raw_text = read_pdf(pdf_path)
                    processed_text = process_text(raw_text)
                    category_texts[category].append(processed_text)

    return category_texts


def read_test_pdf():
    """Read test pdf"""

    pdf_path = os.path.join(Constants.TEST_DOC_DIR_PATH, Constants.TEST_DOC_NAME)
    raw_text = read_pdf(pdf_path)
    processed_text = process_text(raw_text)

    return processed_text
