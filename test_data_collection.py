#!/usr/bin/env python3
"""
Test script for data collection functionality
"""

import os
import sys
import json
from datetime import datetime

def test_reddit_scraper():
    """Test Reddit scraper with a small sample"""
    print("🔴 Testing Reddit scraper...")
    
    cmd = [
        'python', 'src/hmama/train/fetch_reddit_dataset.py',
        '--limit', '5',
        '--output', 'test_reddit.jsonl',
        '--image_dir', 'test_reddit_images',
        '--subs', 'news',
        '--mode', 'hot'
    ]
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Reddit scraper test passed!")
        
        # Check output
        if os.path.exists('test_reddit.jsonl'):
            with open('test_reddit.jsonl', 'r') as f:
                lines = f.readlines()
            print(f"   Collected {len(lines)} Reddit posts")
            
            # Show sample
            if lines:
                sample = json.loads(lines[0])
                print(f"   Sample post: {sample.get('text', '')[:100]}...")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Reddit scraper test failed: {e}")
        return False

def test_newsletter_scraper():
    """Test newsletter scraper with a small sample"""
    print("📰 Testing newsletter scraper...")
    
    cmd = [
        'python', 'src/hmama/train/fetch_newsletter_data.py',
        '--output', 'test_newsletter.jsonl',
        '--image_dir', 'test_newsletter_images',
        '--days_back', '1',
        '--max_articles_per_source', '2'
    ]
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Newsletter scraper test passed!")
        
        # Check output
        if os.path.exists('test_newsletter.jsonl'):
            with open('test_newsletter.jsonl', 'r') as f:
                lines = f.readlines()
            print(f"   Collected {len(lines)} newsletter articles")
            
            # Show sample
            if lines:
                sample = json.loads(lines[0])
                print(f"   Sample article: {sample.get('title', '')[:100]}...")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Newsletter scraper test failed: {e}")
        return False

def test_merge_script():
    """Test dataset merging"""
    print("🔄 Testing dataset merger...")
    
    # Create dummy data if test files don't exist
    if not os.path.exists('test_reddit.jsonl'):
        with open('test_reddit.jsonl', 'w') as f:
            f.write('{"id":"test1","text":"Test reddit post","image":[],"label":0}\n')
    
    if not os.path.exists('test_newsletter.jsonl'):
        with open('test_newsletter.jsonl', 'w') as f:
            f.write('{"id":"test2","text":"Test newsletter article","image":[],"label":1}\n')
    
    cmd = [
        'python', 'src/hmama/train/merge_datasets.py',
        '--reddit_data', 'test_reddit.jsonl',
        '--newsletter_data', 'test_newsletter.jsonl',
        '--output', 'test_merged.jsonl',
        '--max_samples', '10'
    ]
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dataset merger test passed!")
        
        # Check output
        if os.path.exists('test_merged.jsonl'):
            with open('test_merged.jsonl', 'r') as f:
                lines = f.readlines()
            print(f"   Merged {len(lines)} samples")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset merger test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        'test_reddit.jsonl',
        'test_newsletter.jsonl', 
        'test_merged.jsonl'
    ]
    
    test_dirs = [
        'test_reddit_images',
        'test_newsletter_images'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)

def main():
    print("🧪 Testing H-MaMa data collection components...")
    print("=" * 60)
    
    tests = [
        ("Reddit Scraper", test_reddit_scraper),
        ("Newsletter Scraper", test_newsletter_scraper),
        ("Dataset Merger", test_merge_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Data collection is ready to use.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    cleanup_test_files()
    print("✅ Cleanup complete!")

if __name__ == '__main__':
    main()
