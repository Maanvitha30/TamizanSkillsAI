import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import urllib.request
import zipfile

DATA_PATH = 'data/spam.csv'
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

# Download and extract dataset if not present
def download_dataset():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(DATA_PATH):
        print('Downloading dataset...')
        zip_path = 'data/smsspamcollection.zip'
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')
        os.rename('data/SMSSpamCollection', DATA_PATH)
        os.remove(zip_path)
        print('Dataset downloaded and extracted.')

# Load dataset
def load_data():
    download_dataset()
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'text'])
    return df

# Preprocess text
def preprocess(df):
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(df['text'])
    y = (df['label'] == 'spam').astype(int)
    return X, y, vectorizer

# Train/test split, model, evaluation
def main():
    df = load_data()
    X, y, vectorizer = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')

if __name__ == '__main__':
    main() 