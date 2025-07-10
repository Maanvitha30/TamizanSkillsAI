# Email Spam Detection

## Overview
This project builds a lightweight spam classifier using machine learning (Naive Bayes or SVM) to detect spam emails or messages. It preprocesses text, vectorizes using TF-IDF, and evaluates with accuracy, precision, and recall.

## Setup
1. Clone the repository or download the code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) and place it as `data/spam.csv` (or let the script download it automatically).

## Usage
Run the classifier script:
```bash
python spam_classifier.py
```

## Expected Outcome
- 90%+ accuracy spam classifier
- Ready for integration into web forms 