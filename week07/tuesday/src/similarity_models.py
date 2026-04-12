import numpy as np
from typing import List, Tuple
from collections import Counter
import math
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def compute_bow_vector(vocab: List[str], text: str) -> np.ndarray:
    """Computes a strict Bag of Words count vector."""
    tokens = text.lower().split()
    counts = Counter(tokens)
    return np.array([counts.get(w, 0) for w in vocab])

def build_tiny_tfidf(docs: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Builds a tiny TF-IDF space strictly for two documents to illustrate term-matching."""
    tokenized_docs = [d.lower().split() for d in docs]
    vocab = list(set([w for d in tokenized_docs for w in d]))
    
    # Pre-compute IDF
    N = len(docs)
    idf = {}
    for w in vocab:
        df = sum([1 for d in tokenized_docs if w in d])
        idf[w] = math.log((1 + N) / (1 + df)) + 1
        
    vectors = []
    for d in tokenized_docs:
        vec = np.zeros(len(vocab))
        counts = Counter(d)
        for i, w in enumerate(vocab):
            tf = counts.get(w, 0)
            vec[i] = tf * idf[w]
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
        
    return vectors, vocab

def compute_w2v_average(model: Word2Vec, text: str) -> np.ndarray:
    """Averages word2vec embeddings for all tokens in sentence."""
    tokens = text.lower().split()
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def compare_sentences_all_methods(text_a: str, text_b: str, w2v_model: Word2Vec) -> dict:
    """Runs all 4 vector similarity methodologies to measure the semantic gap."""
    results = {}
    
    # Clean split
    ta = text_a.replace(',', '').lower()
    tb = text_b.replace(',', '').lower()
    
    # 1. BOW
    vocab = list(set(ta.split() + tb.split()))
    bow_a = compute_bow_vector(vocab, ta)
    bow_b = compute_bow_vector(vocab, tb)
    
    # calculate cosine
    norm_a, norm_b = np.linalg.norm(bow_a), np.linalg.norm(bow_b)
    if norm_a == 0 or norm_b == 0:
        results["bow"] = 0.0
    else:
        results["bow"] = 1 - cosine(bow_a, bow_b)
        
    # 2. TF-IDF
    tfidf_vecs, _ = build_tiny_tfidf([ta, tb])
    results["tfidf"] = 1 - cosine(tfidf_vecs[0], tfidf_vecs[1])
    
    # 3. Word2Vec Average
    w2v_a = compute_w2v_average(w2v_model, ta)
    w2v_b = compute_w2v_average(w2v_model, tb)
    if np.linalg.norm(w2v_a) == 0 or np.linalg.norm(w2v_b) == 0:
        results["w2v_average"] = 0.0
    else:
        results["w2v_average"] = 1 - cosine(w2v_a, w2v_b)
        
    # 4. Sentence-BERT
    # Loading light all-MiniLM model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_a = sbert_model.encode(text_a)
    sbert_b = sbert_model.encode(text_b)
    results["sentence_bert"] = 1 - cosine(sbert_a, sbert_b)
    
    return results

