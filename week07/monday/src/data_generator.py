import pandas as pd
import numpy as np
import random
import os

def create_dataset():
    np.random.seed(42)
    random.seed(42)
    
    n_reviews = 10000
    categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Beauty', 'Books']
    
    # Base vocabularies
    vocab_electronics = ['wireless', 'earbuds', 'battery', 'life', 'poor', 'excellent', 'charging', 'sound', 'screen', 'the', 'is', 'very']
    vocab_clothing = ['fabric', 'embroidery', 'cotton', 'shirt', 'dress', 'fit', 'size', 'color', 'the', 'is', 'a', 'very']
    vocab_generic = ['good', 'bad', 'okay', 'the', 'is', 'a', 'very', 'product', 'item']
    
    data = []
    
    for i in range(n_reviews):
        cat = random.choice(categories)
        words = []
        
        if cat == 'Electronics':
            # heavily inject 'earbuds' occasionally
            if random.random() < 0.6:
                words.append('earbuds')
            words.extend(random.choices(vocab_electronics, k=random.randint(5, 12)))
        elif cat == 'Clothing':
            words.extend(random.choices(vocab_clothing, k=random.randint(5, 12)))
        else:
            words.extend(random.choices(vocab_generic, k=random.randint(5, 12)))
            
        # Ensure 'the' is everywhere
        words.append('the')
        
        # Make "embroidery" rare overall, even in clothing
        if cat == 'Clothing' and random.random() < 0.05:
            words.append('embroidery')
            
        text = " ".join(words)
        
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
    
    # Specific injection for questions:
    
    # For Q1 b: "wireless earbuds battery life poor" (Top-1 match ideally)
    df.loc[20, 'category'] = 'Electronics'
    df.loc[20, 'review_text'] = 'the wireless earbuds battery life is very poor and bad'
    
    # For Q2: Doc_42 (index 42) must be Clothing and contain 'fabric'
    df.loc[42, 'category'] = 'Clothing'
    df.loc[42, 'review_text'] = 'the fabric of this is great and the embroidery is nice'
    
    # For Q2: IDF('the') should be low (present in almost all)
    # IDF('embroidery') should be high (present in very few)
    
    os.makedirs('week07/monday/data', exist_ok=True)
    df.to_csv('week07/monday/data/ShopSense_Reviews.csv', index=False)
    print("Dataset generated successfully.")

if __name__ == '__main__':
    create_dataset()
