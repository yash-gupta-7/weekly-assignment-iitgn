import pandas as pd
import numpy as np
import random
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_N_REVIEWS = 10000 # 10K rows as per note
DATA_PATH = os.path.join("data", "ShopSense_Reviews_Friday.csv")

def create_friday_reviews(output_path: str = DATA_PATH, n_reviews: int = DEFAULT_N_REVIEWS) -> Optional[pd.DataFrame]:
    """Generates synthetic dataset for Friday assignment tasks with explicit imbalances."""
    try:
        np.random.seed(42)
        random.seed(42)
        
        logger.info(f"Generating {n_reviews} synthetic reviews for Friday final assignment...")
        
        # Category Distribution: Electronics 50%, Clothing 25%, Food 10%, Others 15%
        categories = ['Electronics'] * int(n_reviews * 0.5) + \
                     ['Clothing'] * int(n_reviews * 0.25) + \
                     ['Food'] * int(n_reviews * 0.1) + \
                     ['Home'] * int(n_reviews * 0.05) + \
                     ['Beauty'] * int(n_reviews * 0.05) + \
                     ['Books'] * int(n_reviews * 0.05)
        
        # Catch rounding differences
        while len(categories) < n_reviews:
            categories.append('Books')
        random.shuffle(categories)
        
        # Sentiment Distribution: 5-star (Positive) 70%, 1-star (Negative) 10%, Others (Neutral) 20%
        sentiments = [5] * int(n_reviews * 0.7) + \
                     [1] * int(n_reviews * 0.1) + \
                     [3] * int(n_reviews * 0.2)
        while len(sentiments) < n_reviews:
            sentiments.append(3)
        random.shuffle(sentiments)
        
        # Language Distribution: 15% Code-mixed (Hinglish), 85% English
        languages = ['Code-mixed'] * int(n_reviews * 0.15) + \
                    ['English'] * int(n_reviews * 0.85)
        while len(languages) < n_reviews:
            languages.append('English')
        random.shuffle(languages)
        
        # Corpus content
        pos_words = ["amazing", "great", "excellent", "love", "perfect", "good", "happy"]
        neg_words = ["bad", "terrible", "broke", "waste", "worst", "unhappy", "refund"]
        neut_words = ["okay", "item", "product", "arrived", "shipping", "average"]
        hindi_pos = ["bahut accha", "vadiya", "superb kaam", "m maza aa gaya"]
        hindi_neg = ["bakwas", "bekar quality", "cheated", "paisa barbad"]
        
        data = []
        for i in range(n_reviews):
            lang = languages[i]
            rating = sentiments[i]
            cat = categories[i]
            
            if rating == 5:
                text = " ".join(random.choices(pos_words, k=5))
                if lang == 'Code-mixed': text += " " + random.choice(hindi_pos)
            elif rating == 1:
                text = " ".join(random.choices(neg_words, k=5))
                if lang == 'Code-mixed': text += " " + random.choice(hindi_neg)
            else:
                text = " ".join(random.choices(neut_words, k=5))
                if lang == 'Code-mixed': text += " " + "theek hai"
                
            data.append({
                'review_id': i,
                'category': cat,
                'rating': rating,
                'sentiment_label': 'Positive' if rating == 5 else ('Negative' if rating == 1 else 'Neutral'),
                'language': lang,
                'review_text': text
            })
            
        df = pd.DataFrame(data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset generated successfully and saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        return None

if __name__ == '__main__':
    create_friday_reviews()
