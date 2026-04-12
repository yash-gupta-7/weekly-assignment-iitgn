import pandas as pd
import numpy as np
import random
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_N_REVIEWS = 1000
DATA_PATH = os.path.join("data", "ShopSense_Reviews_Wednesday.csv")

def generate_hard_patterns() -> list:
    """Generates a list of reviews containing hard NLP patterns as specified in Q1."""
    patterns = [
        {"text": "this product is not bad at all", "pattern": "Negation", "expected_sentiment": "Positive"},
        {"text": "Wow great! Broke on day 1", "pattern": "Sarcasm", "expected_sentiment": "Negative"},
        {"text": "Product bahut accha hai lekin delivery late thi", "pattern": "Code-mixing", "expected_sentiment": "Mixed/Negative"},
        {"text": "Returned it within 2 hours", "pattern": "Implicit", "expected_sentiment": "Negative"},
        {"text": "Way better than my previous Samsung", "pattern": "Comparative", "expected_sentiment": "Positive"}
    ]
    return patterns

def generate_aspect_reviews() -> list:
    """Generates reviews with multiple aspect-sentiment pairs for Q2."""
    aspect_reviews = [
        {"text": "Amazing camera quality but the battery is atrocious and customer support was unhelpful.", "category": "Electronics"},
        {"text": "The fabric is soft and comfortable, but the stitching is coming loose after one wash.", "category": "Clothing"},
        {"text": "Delicious food and great ambiance, however the service was extremely slow.", "category": "Food"}
    ]
    return aspect_reviews

def create_wednesday_reviews(output_path: str = DATA_PATH, n_reviews: int = DEFAULT_N_REVIEWS) -> Optional[pd.DataFrame]:
    """Generates synthetic dataset for Wednesday assignment tasks."""
    try:
        np.random.seed(42)
        random.seed(42)
        
        logger.info(f"Generating {n_reviews} synthetic reviews for Wednesday assignment...")
        
        hard_patterns = generate_hard_patterns()
        aspect_reviews = generate_aspect_reviews()
        
        # Add basic noise reviews
        basic_reviews = []
        vocab = ["good", "bad", "okay", "item", "product", "delivery", "price"]
        for i in range(n_reviews - len(hard_patterns) - len(aspect_reviews)):
            text = " ".join(random.choices(vocab, k=random.randint(4, 10)))
            basic_reviews.append({"text": text, "pattern": "Basic", "expected_sentiment": "Neutral"})
            
        all_data = []
        for i, item in enumerate(hard_patterns + aspect_reviews + basic_reviews):
            all_data.append({
                'review_id': i,
                'review_text': item.get('text'),
                'pattern': item.get('pattern', 'None'),
                'category': item.get('category', random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Beauty', 'Books'])),
                'rating': random.randint(1, 5)
            })
            
        df = pd.DataFrame(all_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset generated successfully and saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        return None

if __name__ == '__main__':
    create_wednesday_reviews()
