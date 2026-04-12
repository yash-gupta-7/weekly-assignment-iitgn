import pandas as pd
import numpy as np
import random
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_N_REVIEWS = 1000
DATA_PATH = os.path.join("data", "ShopSense_Reviews_Tuesday.csv")

def generate_polysemous_sentences(n: int) -> list:
    """Generates sentences containing 'cheap' with different meanings."""
    affordable_contexts = [
        "this is a very cheap and affordable option for students",
        "great price so cheap and budget friendly",
        "cheap but high value affordable product",
        "i love how cheap and inexpensive this is",
        "found a cheap deal that is very affordable"
    ]
    flimsy_contexts = [
        "the plastic feels really cheap and flimsy",
        "terrible quality very cheap and easily broken",
        "cheap material that falls apart",
        "so flimsy and cheap i returned it",
        "looks cheap and badly made like a toy"
    ]
    
    sentences = []
    for _ in range(n // 2):
        sentences.append(random.choice(affordable_contexts))
        sentences.append(random.choice(flimsy_contexts))
    return sentences

def generate_random_reviews(n: int) -> list:
    """Generates random noise reviews to build up the Word2Vec corpus."""
    words = ['the', 'is', 'good', 'bad', 'camera', 'phone', 'battery', 'life', 'screen', 'quality', 'great', 'terrible', 'stunning', 'photos', 'drains', 'fast', 'although', 'but', 'incredible']
    sentences = []
    for _ in range(n):
        length = random.randint(5, 12)
        sentences.append(" ".join(random.choices(words, k=length)))
    return sentences

def create_tuesday_reviews(output_path: str = DATA_PATH, n_reviews: int = DEFAULT_N_REVIEWS) -> Optional[pd.DataFrame]:
    """Generates synthetic dataset targeting Word2Vec and Sentence embeddings tasks for Tuesday."""
    try:
        np.random.seed(42)
        random.seed(42)
        
        logger.info(f"Generating {n_reviews} specific synthetic reviews for Tuesday assignment...")
        
        polysemous = generate_polysemous_sentences(n_reviews // 4)
        random_noise = generate_random_reviews(n_reviews - len(polysemous) - 2)
        
        # Inject exact Review A and Review B
        review_a = "incredible camera but terrible battery life"
        review_b = "Battery drains fast, although photos are stunning"
        
        all_reviews = polysemous + random_noise + [review_a, review_b]
        random.shuffle(all_reviews)
        
        data = [{'review_id': i, 'review_text': text} for i, text in enumerate(all_reviews)]
        
        df = pd.DataFrame(data)
        
        # Save to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset generated successfully and saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        return None

if __name__ == '__main__':
    create_tuesday_reviews()
