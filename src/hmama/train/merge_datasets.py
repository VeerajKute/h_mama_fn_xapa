import json
import os
import argparse
from datetime import datetime
import random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--reddit_data', type=str, default='src/hmama/train/reddit_dataset.jsonl')
    p.add_argument('--newsletter_data', type=str, default='data/raw/newsletter_dataset.jsonl')
    p.add_argument('--liar_data', type=str, default='data/processed/LIAR/train.jsonl')
    p.add_argument('--output', type=str, default='data/raw/merged_dataset.jsonl')
    p.add_argument('--reddit_ratio', type=float, default=0.3, help='Ratio of Reddit data in final dataset')
    p.add_argument('--newsletter_ratio', type=float, default=0.4, help='Ratio of newsletter data in final dataset')
    p.add_argument('--liar_ratio', type=float, default=0.3, help='Ratio of LIAR data in final dataset')
    p.add_argument('--max_samples', type=int, default=10000, help='Maximum total samples in merged dataset')
    return p.parse_args()

def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data

def normalize_reddit_data(reddit_data):
    """Normalize Reddit data to match expected format"""
    normalized = []
    for item in reddit_data:
        # Reddit data is typically unlabeled, so we'll use a simple heuristic
        # or mark as unlabeled for manual review
        normalized_item = {
            'id': item.get('id', ''),
            'text': item.get('text', ''),
            'ocr': item.get('ocr', ''),
            'image': item.get('image', []),
            'label': item.get('label', 0),  # Default to 0 (unlabeled)
            'source': 'reddit',
            'subreddit': item.get('subreddit', ''),
            'created_utc': item.get('created_utc', int(datetime.now().timestamp())),
            'url': f"https://reddit.com/r/{item.get('subreddit', '')}/comments/{item.get('id', '')}"
        }
        normalized.append(normalized_item)
    return normalized

def normalize_newsletter_data(newsletter_data):
    """Normalize newsletter data to match expected format"""
    normalized = []
    for item in newsletter_data:
        normalized_item = {
            'id': item.get('id', ''),
            'text': item.get('text', ''),
            'ocr': item.get('ocr', ''),
            'image': item.get('image', []),
            'label': item.get('label', 1),  # Newsletter data is labeled as REAL
            'source': 'newsletter',
            'news_source': item.get('source', ''),
            'category': item.get('category', ''),
            'created_utc': item.get('created_utc', int(datetime.now().timestamp())),
            'url': item.get('url', '')
        }
        normalized.append(normalized_item)
    return normalized

def normalize_liar_data(liar_data):
    """Normalize LIAR data to match expected format"""
    normalized = []
    for item in liar_data:
        # LIAR dataset has different label format, need to convert
        label = 1 if item.get('label', 0) == 1 else 0  # Convert to binary
        normalized_item = {
            'id': item.get('id', ''),
            'text': item.get('text', ''),
            'ocr': item.get('ocr', ''),
            'image': item.get('image', []),
            'label': label,
            'source': 'liar',
            'statement_id': item.get('statement_id', ''),
            'created_utc': item.get('created_utc', int(datetime.now().timestamp())),
            'url': ''
        }
        normalized.append(normalized_item)
    return normalized

def balance_dataset(data, target_ratio, max_samples):
    """Balance dataset according to target ratio"""
    if not data:
        return []
    
    # Calculate how many samples to take
    target_count = int(max_samples * target_ratio)
    actual_count = min(len(data), target_count)
    
    # Randomly sample if we have more data than needed
    if len(data) > actual_count:
        return random.sample(data, actual_count)
    else:
        return data

def merge_datasets():
    args = parse_args()
    
    print("Loading datasets...")
    
    # Load all datasets
    reddit_data = load_jsonl(args.reddit_data)
    newsletter_data = load_jsonl(args.newsletter_data)
    liar_data = load_jsonl(args.liar_data)
    
    print(f"Loaded {len(reddit_data)} Reddit samples")
    print(f"Loaded {len(newsletter_data)} newsletter samples")
    print(f"Loaded {len(liar_data)} LIAR samples")
    
    # Normalize data formats
    print("Normalizing data formats...")
    reddit_normalized = normalize_reddit_data(reddit_data)
    newsletter_normalized = normalize_newsletter_data(newsletter_data)
    liar_normalized = normalize_liar_data(liar_data)
    
    # Balance datasets according to ratios
    print("Balancing datasets...")
    reddit_balanced = balance_dataset(reddit_normalized, args.reddit_ratio, args.max_samples)
    newsletter_balanced = balance_dataset(newsletter_normalized, args.newsletter_ratio, args.max_samples)
    liar_balanced = balance_dataset(liar_normalized, args.liar_ratio, args.max_samples)
    
    print(f"Balanced: {len(reddit_balanced)} Reddit, {len(newsletter_balanced)} newsletter, {len(liar_balanced)} LIAR")
    
    # Merge datasets
    merged_data = reddit_balanced + newsletter_balanced + liar_balanced
    
    # Shuffle the merged dataset
    random.shuffle(merged_data)
    
    # Add metadata
    for i, item in enumerate(merged_data):
        item['merged_id'] = f"merged_{i:06d}"
        item['merge_timestamp'] = int(datetime.now().timestamp())
    
    # Save merged dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\nMerged dataset saved to {args.output}")
    print(f"Total samples: {len(merged_data)}")
    
    # Count by source
    source_counts = {}
    label_counts = {0: 0, 1: 0}
    
    for item in merged_data:
        source = item.get('source', 'unknown')
        label = item.get('label', 0)
        source_counts[source] = source_counts.get(source, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    print("\nLabel distribution:")
    print(f"  FAKE (0): {label_counts[0]}")
    print(f"  REAL (1): {label_counts[1]}")
    
    # Count samples with images
    with_images = sum(1 for item in merged_data if item.get('image', []))
    print(f"\nSamples with images: {with_images}")

if __name__ == '__main__':
    merge_datasets()
