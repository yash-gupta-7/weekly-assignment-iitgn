import math
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Tokenizes a string into a list of lowercase words."""
    if not isinstance(text, str):
        return []
    return text.lower().split()

def compute_corpus_stats(docs: List[str]) -> Tuple[List[Counter], Counter, Dict[str, float]]:
    """
    Computes term frequencies, document frequencies, and IDFs for a corpus.
    
    Args:
        docs (List[str]): List of document strings.
        
    Returns:
        Tuple containing list of term frequencies per doc, global document frequency, and IDF mapping.
    """
    logger.info("Computing corpus term frequencies and document frequencies...")
    tf_list = []
    df_counts = Counter()
    
    for doc in docs:
        tokens = tokenize(doc)
        counts = Counter(tokens)
        tf_list.append(counts)
        for term in set(tokens):
            df_counts[term] += 1
            
    N = len(docs)
    idf = {}
    for term, count in df_counts.items():
        # Scikit-learn default IDF smoothing
        idf[term] = math.log((1 + N) / (1 + count)) + 1
        
    return tf_list, df_counts, idf

def build_tfidf_matrix(tf_list: List[Counter], idf: Dict[str, float], vocab: List[str]) -> sp.csr_matrix:
    """
    Builds an L2-normalized sparse TF-IDF matrix.
    
    Args:
        tf_list: List of Counter objects for each document's term frequencies.
        idf: Dictionary mapping term to IDF score.
        vocab: Sorted list of vocabulary words.
        
    Returns:
        SciPy CSR Sparse Matrix of shape (len(tf_list), len(vocab)).
    """
    logger.info("Building sparse TF-IDF matrix...")
    N = len(tf_list)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    tfidf_matrix = sp.lil_matrix((N, len(vocab)), dtype=np.float32)
    
    for i, tf_dict in enumerate(tf_list):
        if sum(tf_dict.values()) == 0:
            continue
            
        row_norm = 0.0
        # First compute Euclidean norm
        for term, count in tf_dict.items():
            if term in vocab_index:
                tfidf_val = count * idf[term]
                row_norm += tfidf_val ** 2
        row_norm = math.sqrt(row_norm) if row_norm > 0 else 1.0
        
        # Assign L2 normalized values
        for term, count in tf_dict.items():
            if term in vocab_index:
                j = vocab_index[term]
                tfidf_val = count * idf[term]
                tfidf_matrix[i, j] = tfidf_val / row_norm
                
    return tfidf_matrix.tocsr()

def compute_cosine_similarity(query: str, tfidf_matrix: sp.csr_matrix, idf: Dict[str, float], vocab_index: Dict[str, int]) -> np.ndarray:
    """
    Computes cosine similarity between a text query and the TF-IDF matrix.
    
    Args:
        query: Query string.
        tfidf_matrix: Document matrix (L2 normalized).
        idf: IDF dictionary map.
        vocab_index: Vocabulary index map.
        
    Returns:
        1D numpy array of cosine similarity scores.
    """
    query_tokens = tokenize(query)
    query_tf = Counter(query_tokens)
    query_vec = np.zeros(len(vocab_index))
    
    for term, count in query_tf.items():
        if term in vocab_index:
            query_vec[vocab_index[term]] = count * idf[term]
            
    q_norm = np.linalg.norm(query_vec)
    if q_norm > 0:
        query_vec = query_vec / q_norm
        
    return tfidf_matrix.dot(query_vec)
