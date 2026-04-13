import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from typing import Dict, Any, Tuple

def train_and_evaluate_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Trains two models (LogReg and Naive Bayes) and returns performance metrics."""
    
    # Preprocessing
    df['text'] = df['review_text'].fillna('')
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment_label'], test_size=0.2, random_state=42, stratify=df['sentiment_label']
    )
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    results = {}
    
    # Model 1: Logistic Regression (Fast)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    
    results['LR'] = {
        'model': lr,
        'f1': f1_score(y_test, y_pred_lr, average='weighted'),
        'report': classification_report(y_test, y_pred_lr, output_dict=True),
        'vectorizer': vectorizer
    }
    
    # Model 2: Naive Bayes (Standard Baseline)
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    
    results['NB'] = {
        'model': nb,
        'f1': f1_score(y_test, y_pred_nb, average='weighted'),
        'report': classification_report(y_test, y_pred_nb, output_dict=True)
    }
    
    return results, X_test, y_test

def get_priya_summary(results: Dict[str, Any]) -> str:
    """Generates a plain-language summary for non-technical stakeholder Priya."""
    lr_f1 = results['LR']['f1']
    nb_f1 = results['NB']['f1']
    
    best_model = "Logistic Regression" if lr_f1 > nb_f1 else "Naive Bayes"
    f1_val = max(lr_f1, nb_f1)
    
    summary = f"""
    --- Performance Summary for Priya (Non-Technical) ---
    Our testing shows that the '{best_model}' model is the most reliable for ShopSense.
    
    What the numbers mean for our business:
    1. Overall Effectiveness (F1-score: {f1_val:.2f}): This number tells us how well the model balances finding 
       actual angry customers without bothering happy ones by mistake. A 1.0 would be perfect; our current 
       score indicates we are capturing most sentiment correctly, but there's room for improvement in niche categories.
       
    2. Missing Complaints (Recall): The model correctly identifies roughly {results['LR']['report']['Negative']['recall']:.0%} 
       of 1-star reviews. This means we might 'miss' about {1 - results['LR']['report']['Negative']['recall']:.0%} of unhappy 
       customers in real-time, who would then need to be caught by our monitoring systems.
       
    3. False Alarms (Precision): When the model says a review is 'Negative', it is correct {results['LR']['report']['Negative']['precision']:.0%} 
       of the time. The remaining {1 - results['LR']['report']['Negative']['precision']:.0%} are false alarms where the customer 
       was actually okay, but we might have flagged them for a refund unnecessarily.
    """
    return summary
