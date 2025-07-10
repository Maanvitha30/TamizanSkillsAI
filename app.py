from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

# Global variables for the model
vectorizer = None
model = None

def load_or_train_model():
    global vectorizer, model
    
    # Check if model files exist
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        # Load existing model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model loaded from files")
    else:
        # Train new model
        print("Training new model...")
        from spam_classifier import load_data, preprocess
        
        # Load and preprocess data
        df = load_data()
        X, y, vectorizer = preprocess(df)
        
        # Train model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Save model and vectorizer
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print("Model trained and saved")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Transform the text using the same vectorizer
        X = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Get confidence scores
        spam_prob = probability[1] if prediction == 1 else probability[0]
        ham_prob = probability[0] if prediction == 0 else probability[1]
        
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': {
                'spam': round(spam_prob * 100, 2),
                'ham': round(ham_prob * 100, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_or_train_model()
    app.run(debug=True, host='0.0.0.0', port=5000) 