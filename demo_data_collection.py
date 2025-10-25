#!/usr/bin/env python3
"""
Demo script showing how to use the H-MaMa data collection system
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_demo():
    print_header("H-MaMa Data Collection Demo")
    print("This demo shows how to collect and validate data for the H-MaMa fake news detection system.")
    
    # Step 1: Test individual components
    print_step(1, "Testing Individual Components")
    
    print("🔴 Testing Reddit scraper...")
    reddit_cmd = [
        'python', 'src/hmama/train/fetch_reddit_dataset.py',
        '--limit', '5',
        '--output', 'demo_reddit.jsonl',
        '--image_dir', 'demo_reddit_images',
        '--subs', 'news',
        '--mode', 'hot'
    ]
    
    try:
        result = subprocess.run(reddit_cmd, check=True, capture_output=True, text=True)
        print("✅ Reddit scraper test passed!")
        
        # Show sample data
        if os.path.exists('demo_reddit.jsonl'):
            with open('demo_reddit.jsonl', 'r') as f:
                lines = f.readlines()
            print(f"   Collected {len(lines)} Reddit posts")
            
            if lines:
                sample = json.loads(lines[0])
                print(f"   Sample: {sample.get('text', '')[:100]}...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Reddit scraper test failed: {e}")
        return False
    
    print("\n📰 Testing newsletter scraper...")
    newsletter_cmd = [
        'python', 'src/hmama/train/fetch_newsletter_data.py',
        '--output', 'demo_newsletter.jsonl',
        '--image_dir', 'demo_newsletter_images',
        '--days_back', '1',
        '--max_articles_per_source', '2'
    ]
    
    try:
        result = subprocess.run(newsletter_cmd, check=True, capture_output=True, text=True)
        print("✅ Newsletter scraper test passed!")
        
        # Show sample data
        if os.path.exists('demo_newsletter.jsonl'):
            with open('demo_newsletter.jsonl', 'r') as f:
                lines = f.readlines()
            print(f"   Collected {len(lines)} newsletter articles")
            
            if lines:
                sample = json.loads(lines[0])
                print(f"   Sample: {sample.get('title', '')[:100]}...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Newsletter scraper test failed: {e}")
        return False
    
    # Step 2: Merge datasets
    print_step(2, "Merging Datasets")
    
    print("🔄 Testing dataset merger...")
    merge_cmd = [
        'python', 'src/hmama/train/merge_datasets.py',
        '--reddit_data', 'demo_reddit.jsonl',
        '--newsletter_data', 'demo_newsletter.jsonl',
        '--output', 'demo_merged.jsonl',
        '--max_samples', '20'
    ]
    
    try:
        result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
        print("✅ Dataset merger test passed!")
        
        # Show merged data stats
        if os.path.exists('demo_merged.jsonl'):
            with open('demo_merged.jsonl', 'r') as f:
                lines = f.readlines()
            print(f"   Merged {len(lines)} samples")
            
            # Analyze sources
            sources = {}
            labels = {}
            for line in lines:
                item = json.loads(line)
                source = item.get('source', 'unknown')
                label = item.get('label', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                labels[label] = labels.get(label, 0) + 1
            
            print("   Source distribution:")
            for source, count in sources.items():
                print(f"     {source}: {count}")
            
            print("   Label distribution:")
            for label, count in labels.items():
                print(f"     {label}: {count}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset merger test failed: {e}")
        return False
    
    # Step 3: Validate data
    print_step(3, "Validating Data Quality")
    
    print("🔍 Testing data validation...")
    validate_cmd = [
        'python', 'src/hmama/train/validate_data.py',
        '--dataset', 'demo_merged.jsonl',
        '--output', 'demo_validation_report.json',
        '--check_images',
        '--check_duplicates'
    ]
    
    try:
        result = subprocess.run(validate_cmd, check=True, capture_output=True, text=True)
        print("✅ Data validation test passed!")
        
        # Show validation results
        if os.path.exists('demo_validation_report.json'):
            with open('demo_validation_report.json', 'r') as f:
                report = json.load(f)
            
            print(f"   Total items: {report['total_items']}")
            print(f"   Total issues: {report['summary']['total_issues']}")
            print(f"   Data quality score: {report['summary']['data_quality_score']}/100")
    except subprocess.CalledProcessError as e:
        print(f"❌ Data validation test failed: {e}")
        return False
    
    # Step 4: Show how to use in training
    print_step(4, "Using Data in Training")
    
    print("🧠 The collected data can now be used for training:")
    print("   python src/hmama/train/train_content.py --dataset demo_merged.jsonl")
    print("   python src/hmama/train/train_full.py --dataset demo_merged.jsonl")
    
    # Step 5: Cleanup
    print_step(5, "Cleanup")
    
    print("🧹 Cleaning up demo files...")
    demo_files = [
        'demo_reddit.jsonl',
        'demo_newsletter.jsonl',
        'demo_merged.jsonl',
        'demo_validation_report.json'
    ]
    
    demo_dirs = [
        'demo_reddit_images',
        'demo_newsletter_images'
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   Removed {file}")
    
    for dir_name in demo_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"   Removed directory {dir_name}")
    
    print("✅ Cleanup complete!")
    
    # Final summary
    print_header("Demo Complete!")
    print("🎉 All data collection components are working correctly!")
    print("\n📚 Next steps:")
    print("1. Run the full data collection: python src/hmama/train/collect_all_data.py")
    print("2. Validate your data: python src/hmama/train/validate_data.py --dataset data/raw/merged_dataset.jsonl")
    print("3. Train your model with the collected data")
    print("4. Set up regular data collection for fresh content")
    
    return True

if __name__ == '__main__':
    success = run_demo()
    if success:
        print("\n✅ Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo failed. Check the output above for details.")
        sys.exit(1)
