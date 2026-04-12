import pandas as pd
import numpy as np
import random
import os
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_N_REVIEWS = 10000
CATEGORIES = ['Electronics', 'Clothing', 'Food', 'Home', 'Beauty', 'Books']
DATA_PATH = os.path.join("data", "ShopSense_Reviews.csv")

def generate_review_text(category: str) -> str:
    """Generates synthetic review text based on the product category."""
    vocab_electronics = ['wireless', 'earbuds', 'battery', 'life', 'poor', 'excellent', 'charging', 'sound', 'screen', 'the', 'is', 'very']
    vocab_clothing = ['fabric', 'embroidery', 'cotton', 'shirt', 'dress', 'fit', 'size', 'color', 'the', 'is', 'a', 'very']
    vocab_generic = ['good', 'bad', 'okay', 'the', 'is', 'a', 'very', 'product', 'item']
    
    words = []
    if category == 'Electronics':
        if random.random() < 0.6:
            words.append('earbuds')
        words.extend(random.choices(vocab_electronics, k=random.randint(5, 12)))
    elif category == 'Clothing':
        words.extend(random.choices(vocab_clothing, k=random.randint(5, 12)))
    else:
        words.extend(random.choices(vocab_generic, k=random.randint(5, 12)))
        
    words.append('the')
    
    if category == 'Clothing' and random.random() < 0.05:
        words.append('embroidery')
        
    return " ".join(words)

def create_shopsense_reviews(output_path: str = DATA_PATH, n_reviews: int = DEFAULT_N_REVIEWS) -> Optional[pd.DataFrame]:
    """
    Generates a synthetic dataset of ShopSense reviews and saves it to a CSV file.
    
    Args:
        output_path (str): The file path where the CSV will be saved.
        n_reviews (int): Number of reviews to generate.
        
    Returns:
        pd.DataFrame: The generated review dataset, or None if generation failed.
    """
    try:
        np.random.seed(42)
        random.seed(42)
        
        logger.info(f"Generating {n_reviews} synthetic reviews...")
        data = []
        for i in range(n_reviews):
            cat = random.choice(CATEGORIES)
            text = generate_review_text(cat)
            
            row = {
                'review_id': i,
                'customer_id': random.randint(1000, 9999),
                'product_id': random.randint(100, 999),
                'category': cat,
                'review_text': text,
                'rating': random.randint(1, 5)
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Inject specific targets for Q1 and Q2
        df.loc[20, 'category'] = 'Electronics'
        df.loc[20, 'review_text'] = 'the wireless earbuds battery life is very poor and bad'
        
        df.loc[42, 'category'] = 'Clothing'
        df.loc[42, 'review_text'] = 'the fabric of this is great and the embroidery is nice'
        
        # Save to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset generated successfully and saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        return None

if __name__ == '__main__':
    create_shopsense_reviews()
