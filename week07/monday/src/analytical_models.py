import math
from typing import List, Dict, Optional
from collections import Counter
import numpy as np

def compute_manual_tfidf(term: str, doc_tokens: List[str], N: int, df_count: int) -> dict:
    """
    Computes manual TF, IDF, and TF-IDF for a specific term and document.
    
    Args:
        term: The target word.
        doc_tokens: List of words in the specific document.
        N: Total number of documents.
        df_count: Document frequency for the term.
        
    Returns:
        Dictionary with calculation results.
    """
    tf = doc_tokens.count(term)
    idf = math.log((1 + N) / (1 + df_count)) + 1
    tfidf_raw = tf * idf
    
    return {
        'tf': tf,
        'idf': idf,
        'tfidf_raw': tfidf_raw
    }

def compute_bm25_scores(docs: List[str], query_tokens: List[str], df_counts: Counter, N: int, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """
    Computes BM25 similarities for a given query over the entire corpus.
    
    Args:
        docs: List of full corpus documents.
        query_tokens: List of query words.
        df_counts: Global document frequency map.
        N: Total document count.
        k1: Saturation parameter.
        b: Document length normalization parameter.
        
    Returns:
        1D numpy array of BM25 scores matching doc index.
    """
    tokenized_docs = [d.lower().split() for d in docs]
    avgdl = sum(len(d) for d in tokenized_docs) / N
    
    bm25_scores = []
    
    for doc_tokens in tokenized_docs:
        score = 0.0
        dl = len(doc_tokens)
        for q in query_tokens:
            if q not in df_counts: 
                continue
                
            tf = doc_tokens.count(q)
            # BM25 IDF formula
            idf_bm25 = math.log((N - df_counts[q] + 0.5) / (df_counts[q] + 0.5) + 1)
            
            term_freq = tf * (k1 + 1)
            norm = tf + k1 * (1 - b + b * (dl / avgdl))
            
            score += idf_bm25 * (term_freq / norm)
            
        bm25_scores.append(score)
        
    return np.array(bm25_scores)
