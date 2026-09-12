import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def load_and_prep_dataset():
    print("Loading dataset from HuggingFace...")
    # 'gorkemsevinc/Customer_Support_on_Twitter' has ~2.81M rows
    dataset = load_dataset('gorkemsevinc/Customer_Support_on_Twitter', split='train')
    
    print("Converting to pandas...")
    df = dataset.to_pandas()
    
    print("Analyzing brands...")
    top_brands = df['author_id'].value_counts().head(20)
    print("Top Brands by number of tweets:")
    print(top_brands)

    # Pick AppleSupport as default example since it usually handles tons of IT/device queries
    # AppleSupport is a good representative of a "brand" resolving issues.
    brand = 'AppleSupport'
    
    # Filter AppleSupport and their customer interactions.
    # The dataset has 'tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'response_tweet_id', 'in_response_to_tweet_id'
    
    print(f"\nFiltering for {brand}...")
    brand_df = df[df['author_id'] == brand]
    
    # We want customer messages that AppleSupport replied to.
    print(f"Number of tweets by {brand}: {len(brand_df)}")
    
    # Save a sample to quickly look at it.
    sample = brand_df.head(50)
    sample.to_csv('sample_brand_tweets.csv', index=False)
    print("Sample saved to sample_brand_tweets.csv")

if __name__ == "__main__":
    load_and_prep_dataset()
