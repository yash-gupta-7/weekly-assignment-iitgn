from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import numpy as np
from typing import List, Dict, Tuple

def train_word2vec(sentences: List[str], window_size: int = 5) -> Word2Vec:
    """Trains Word2Vec model on pre-tokenized sentences."""
    tokenized = [s.lower().split() for s in sentences]
    model = Word2Vec(sentences=tokenized, vector_size=50, window=window_size, min_count=1, workers=1, seed=42)
    return model

def get_word_vector(model: Word2Vec, word: str) -> np.ndarray:
    """Returns vector for a word, or zeros if missing."""
    if word in model.wv:
        return model.wv[word]
    return np.zeros(model.vector_size)

def compute_cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return 1 - cosine(v1, v2)

def compare_polysemous_similarity(model: Word2Vec) -> Dict[str, float]:
    """Compares 'cheap' against 'affordable' and 'flimsy'."""
    v_cheap = get_word_vector(model, "cheap")
    v_affordable = get_word_vector(model, "affordable")
    v_flimsy = get_word_vector(model, "flimsy")
    
    return {
        "cheap_vs_affordable": compute_cosine_sim(v_cheap, v_affordable),
        "cheap_vs_flimsy": compute_cosine_sim(v_cheap, v_flimsy)
    }

def disambiguate_context(sentence: str, model: Word2Vec) -> str:
    """Classifies the meaning of 'cheap' in a sentence using context averaging."""
    words = sentence.lower().split()
    if 'cheap' not in words:
        return "Unknown"
        
    context_words = [w for w in words if w != 'cheap']
    if not context_words:
        return "Ambiguous"
        
    context_vectors = [get_word_vector(model, w) for w in context_words if w in model.wv]
    if not context_vectors:
        return "Ambiguous"
        
    context_avg = np.mean(context_vectors, axis=0)
    
    sim_affordable = compute_cosine_sim(context_avg, get_word_vector(model, "affordable"))
    sim_flimsy = compute_cosine_sim(context_avg, get_word_vector(model, "flimsy"))
    
    if sim_affordable > sim_flimsy:
        return "Affordable"
    else:
        return "Low-Quality (Flimsy)"

