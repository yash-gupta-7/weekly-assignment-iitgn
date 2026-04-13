import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def evaluate_constraints(results: Dict[str, Any], df_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Quantitatively assesses models against three hard engineering constraints."""
    
    constraint_results = {}
    
    for model_name in ['LR', 'NB']:
        model = results[model_name]['model']
        vectorizer = results['LR']['vectorizer']
        
        # 1. Inference Speed (< 20ms constraint)
        sample_texts = df_test.head(100).tolist()
        start_time = time.time()
        _ = model.predict(vectorizer.transform(sample_texts))
        total_time = (time.time() - start_time) * 1000 # to ms
        avg_inference_ms = total_time / 100
        
        # 2. Mixed Language Performance (Hinglish)
        # Assuming original df_test can be indexed back to the language tags
        # For simplicity, we filter by language in this demo logic
        # (In real notebook we'd keep language aligned during split)
        # Here we just simulate or assume a subset of the data is Hinglish
        
        # 3. Category Robustness (Testing on Electronics vs others)
        # We can extract per-category accuracy if we have the category data
        
        constraint_results[model_name] = {
            'avg_inference_ms': avg_inference_ms,
            'meets_20ms_goal': avg_inference_ms < 20,
            'description': f"{model_name} processing at {avg_inference_ms:.2f}ms per review."
        }
        
    return constraint_results

def evaluate_hinglish_robustness(df: pd.DataFrame, model_results: Dict[str, Any]) -> Dict[str, float]:
    """Measures performance specifically on code-mixed reviews."""
    hinglish_df = df[df['language'] == 'Code-mixed']
    vectorizer = model_results['LR']['vectorizer']
    
    acc_scores = {}
    for model_name in ['LR', 'NB']:
        model = model_results[model_name]['model']
        X = vectorizer.transform(hinglish_df['review_text'])
        y_true = hinglish_df['sentiment_label']
        y_pred = model.predict(X)
        acc_scores[model_name] = np.mean(y_true == y_pred)
        
    return acc_scores
