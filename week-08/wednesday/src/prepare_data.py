
import pandas as pd
import numpy as np
import os

def create_social_media_dataset(reddit_path, twitter_path, output_path):
    # Load data
    reddit_df = pd.read_csv(reddit_path).rename(columns={'clean_comment': 'text'})
    twitter_df = pd.read_csv(twitter_path).rename(columns={'clean_text': 'text'})
    
    # Add platform
    reddit_df['platform'] = 'Reddit'
    twitter_df['platform'] = 'Twitter'
    
    # Sample 1500 from each to get 3000 rows
    reddit_sample = reddit_df.sample(n=1500, random_state=42)
    twitter_sample = twitter_df.sample(n=1500, random_state=42)
    
    # Combine
    df = pd.concat([reddit_sample, twitter_sample], ignore_index=True)
    
    # Add fake columns to match assignment description
    df['language'] = 'English' # Mostly English in these datasets
    
    # Simulate topics
    topics = ['Politics', 'Gadgets', 'Social Issues', 'Sports', 'Miscellaneous']
    df['topic'] = np.random.choice(topics, size=len(df))
    
    # Map category to sentiment
    # Original: -1 (Negative), 0 (Neutral), 1 (Positive)
    df = df.rename(columns={'category': 'sentiment'})
    
    # Add hate_speech and spam flags (highly imbalanced as per instructions)
    # Hate speech: mostly from negative sentiment, but rare
    df['hate_speech'] = 0
    # Flag ~5% as hate speech if sentiment is negative
    neg_indices = df[df['sentiment'] == -1].index
    hate_indices = np.random.choice(neg_indices, size=int(len(neg_indices) * 0.15), replace=False)
    df.loc[hate_indices, 'hate_speech'] = 1
    
    # Spam: random, very rare (~2%)
    df['spam'] = 0
    spam_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[spam_indices, 'spam'] = 1
    
    # Add some noise/data quality issues (missing values)
    df.loc[np.random.choice(df.index, 20), 'text'] = np.nan
    df.loc[np.random.choice(df.index, 10), 'sentiment'] = np.nan
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    base_path = "week-08/wednesday/data"
    create_social_media_dataset(
        os.path.join(base_path, "Reddit_Data.csv"),
        os.path.join(base_path, "Twitter_Data.csv"),
        os.path.join(base_path, "social_media_posts.csv")
    )
