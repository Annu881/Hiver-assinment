import os
import pandas as pd
from datasets import load_dataset
import json

def extract_brand_dataset():
    print("Loading conversation dataset...")
    # Loading TNE-AI dataset which contains pre-grouped conversations
    dataset = load_dataset('TNE-AI/customer-support-on-twitter-conversation', split='train')
    df = dataset.to_pandas()
    
    brand = 'AppleSupport'
    print(f"Total conversations in dataset: {len(df)}")
    
    # Filter for AppleSupport
    brand_df = df[df['company'] == brand]
    print(f"Total {brand} conversations: {len(brand_df)}")
    
    # We need a sample of 250 for Golden Evaluation Set, maybe leave some extra for training/exploration
    exploration_sample = brand_df.head(250)
    
    # Parse the strings representing conversations into JSON/dicts since they are likely stringified lists/dicts
    def parse_conversation(conv_str):
        try:
            # Usually these are single quotes inside string representing list of dicts, let's fix to double
            # Or use eval if it's a python dict literal
            import ast
            return ast.literal_eval(conv_str)
        except Exception as e:
            return conv_str
            
    exploration_sample['parsed_conversation'] = exploration_sample['conversation'].apply(parse_conversation)
    
    # Save the sample
    exploration_sample.to_json('apple_support_sample.json', orient='records', indent=2)
    print("Saved 250 sample conversations to apple_support_sample.json")

    # Let's print out the first conversation to see the structure
    first_conv = exploration_sample.iloc[0]['parsed_conversation']
    print("\nFirst conversation format:")
    print(json.dumps(first_conv, indent=2))
    
if __name__ == "__main__":
    extract_brand_dataset()
