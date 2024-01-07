import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Constants import Constants
from DataPreprocessingUtils import read_all_train_pdfs, read_test_pdf


def extract_vocabulary(data):
    """Extract list of unique words form the train PDFs"""

    documents = []

    for key in data.keys():
        for document in data[key]:
            documents.append(document)

    master_document = ' '.join(documents)
    tokens = nltk.word_tokenize(master_document)
    vocabulary = set(tokens)

    return vocabulary, documents


def create_term_document_table(vocabulary, documents):
    """Create the term-document table using the train PDFs"""

    vectorizer = CountVectorizer(vocabulary=vocabulary)
    vectorizer_output = vectorizer.fit_transform(documents)

    term_document_matrix = pd.DataFrame(vectorizer_output.toarray(), columns=vectorizer.get_feature_names_out())

    return vectorizer, term_document_matrix


def predict_test_document_category(vectorizer, term_document_matrix):
    """Predict the category of test document using model created"""

    test_document = read_test_pdf()
    test_vector = vectorizer.transform([test_document]).toarray()

    cos_similarities = cosine_similarity(test_vector, term_document_matrix)
    log_cos_similarities = np.log1p(cos_similarities)
    most_similar_doc_index = np.argmax(log_cos_similarities)

    category_index = most_similar_doc_index // Constants.NUM_DOCS_PER_CATEGORY
    predicted_category = Constants.CATEGORIES[category_index]

    return predicted_category


def train_and_run_model():
    """Train the model and use it to predict category of test document"""

    data = read_all_train_pdfs()
    vocabulary, documents = extract_vocabulary(data)
    vectorizer, term_document_matrix = create_term_document_table(vocabulary, documents)
    predicted_category = predict_test_document_category(vectorizer, term_document_matrix)

    return predicted_category
