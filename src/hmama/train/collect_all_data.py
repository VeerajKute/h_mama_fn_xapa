#!/usr/bin/env python3
"""
Comprehensive data collection script for H-MaMa fake news detection system.
This script collects data from multiple sources and creates a balanced training dataset.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

def parse_args():
    p = argparse.ArgumentParser(description='Collect data from all sources for H-MaMa training')
    p.add_argument('--reddit_limit', type=int, default=200, help='Number of Reddit posts to collect')
    p.add_argument('--newsletter_days', type=int, default=7, help='Days back to fetch newsletter data')
    p.add_argument('--max_articles_per_source', type=int, default=30, help='Max articles per newsletter source')
    p.add_argument('--output_dir', type=str, default='data/raw', help='Output directory for all data')
    p.add_argument('--skip_reddit', action='store_true', help='Skip Reddit data collection')
    p.add_argument('--skip_newsletter', action='store_true', help='Skip newsletter data collection')
    p.add_argument('--skip_merge', action='store_true', help='Skip dataset merging')
    p.add_argument('--clean', action='store_true', help='Clean existing data before collection')
    return p.parse_args()

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def clean_data(output_dir):
    """Clean existing data files"""
    print("🧹 Cleaning existing data...")
    
    files_to_clean = [
        'reddit_dataset.jsonl',
        'newsletter_dataset.jsonl',
        'merged_dataset.jsonl'
    ]
    
    dirs_to_clean = [
        'reddit_images',
        'newsletter_images'
    ]
    
    for file in files_to_clean:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
    
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(output_dir, dir_name)
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"Removed directory {dir_path}")

def collect_reddit_data(output_dir, reddit_limit):
    """Collect data from Reddit"""
    print("\n🔴 Collecting Reddit data...")
    
    reddit_output = os.path.join(output_dir, 'reddit_dataset.jsonl')
    reddit_images = os.path.join(output_dir, 'reddit_images')
    
    cmd = [
        'python', 'src/hmama/train/fetch_reddit_dataset.py',
        '--limit', str(reddit_limit),
        '--output', reddit_output,
        '--image_dir', reddit_images,
        '--mode', 'hot',
        '--max_images_per_post', '2'
    ]
    
    return run_command(cmd, "Reddit data collection")

def collect_newsletter_data(output_dir, newsletter_days, max_articles_per_source):
    """Collect data from newsletter sources"""
    print("\n📰 Collecting newsletter data...")
    
    newsletter_output = os.path.join(output_dir, 'newsletter_dataset.jsonl')
    newsletter_images = os.path.join(output_dir, 'newsletter_images')
    
    cmd = [
        'python', 'src/hmama/train/fetch_newsletter_data.py',
        '--output', newsletter_output,
        '--image_dir', newsletter_images,
        '--days_back', str(newsletter_days),
        '--max_articles_per_source', str(max_articles_per_source)
    ]
    
    return run_command(cmd, "Newsletter data collection")

def merge_datasets(output_dir):
    """Merge all datasets"""
    print("\n🔄 Merging datasets...")
    
    reddit_data = os.path.join(output_dir, 'reddit_dataset.jsonl')
    newsletter_data = os.path.join(output_dir, 'newsletter_dataset.jsonl')
    liar_data = 'data/processed/LIAR/train.jsonl'
    merged_output = os.path.join(output_dir, 'merged_dataset.jsonl')
    
    cmd = [
        'python', 'src/hmama/train/merge_datasets.py',
        '--reddit_data', reddit_data,
        '--newsletter_data', newsletter_data,
        '--liar_data', liar_data,
        '--output', merged_output,
        '--reddit_ratio', '0.3',
        '--newsletter_ratio', '0.4',
        '--liar_ratio', '0.3',
        '--max_samples', '10000'
    ]
    
    return run_command(cmd, "Dataset merging")

def generate_data_report(output_dir):
    """Generate a data collection report"""
    print("\n📊 Generating data report...")
    
    report = {
        'collection_timestamp': datetime.now().isoformat(),
        'output_directory': output_dir,
        'datasets': {}
    }
    
    # Check Reddit data
    reddit_file = os.path.join(output_dir, 'reddit_dataset.jsonl')
    if os.path.exists(reddit_file):
        with open(reddit_file, 'r', encoding='utf-8') as f:
            reddit_count = sum(1 for line in f if line.strip())
        report['datasets']['reddit'] = {
            'file': reddit_file,
            'count': reddit_count,
            'status': 'collected'
        }
    
    # Check newsletter data
    newsletter_file = os.path.join(output_dir, 'newsletter_dataset.jsonl')
    if os.path.exists(newsletter_file):
        with open(newsletter_file, 'r', encoding='utf-8') as f:
            newsletter_count = sum(1 for line in f if line.strip())
        report['datasets']['newsletter'] = {
            'file': newsletter_file,
            'count': newsletter_count,
            'status': 'collected'
        }
    
    # Check merged data
    merged_file = os.path.join(output_dir, 'merged_dataset.jsonl')
    if os.path.exists(merged_file):
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_count = sum(1 for line in f if line.strip())
        report['datasets']['merged'] = {
            'file': merged_file,
            'count': merged_count,
            'status': 'merged'
        }
    
    # Save report
    report_file = os.path.join(output_dir, 'data_collection_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Data collection report saved to {report_file}")
    
    # Print summary
    print("\n📈 Collection Summary:")
    for dataset, info in report['datasets'].items():
        print(f"  {dataset}: {info['count']} samples - {info['status']}")

def main():
    args = parse_args()
    
    print("🚀 Starting H-MaMa data collection...")
    print(f"Output directory: {args.output_dir}")
    print(f"Reddit limit: {args.reddit_limit}")
    print(f"Newsletter days back: {args.newsletter_days}")
    print(f"Max articles per source: {args.max_articles_per_source}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean existing data if requested
    if args.clean:
        clean_data(args.output_dir)
    
    success_count = 0
    total_steps = 0
    
    # Collect Reddit data
    if not args.skip_reddit:
        total_steps += 1
        if collect_reddit_data(args.output_dir, args.reddit_limit):
            success_count += 1
    else:
        print("⏭️ Skipping Reddit data collection")
    
    # Collect newsletter data
    if not args.skip_newsletter:
        total_steps += 1
        if collect_newsletter_data(args.output_dir, args.newsletter_days, args.max_articles_per_source):
            success_count += 1
    else:
        print("⏭️ Skipping newsletter data collection")
    
    # Merge datasets
    if not args.skip_merge:
        total_steps += 1
        if merge_datasets(args.output_dir):
            success_count += 1
    else:
        print("⏭️ Skipping dataset merging")
    
    # Generate report
    generate_data_report(args.output_dir)
    
    # Final summary
    print(f"\n🎯 Data collection completed!")
    print(f"Successfully completed {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("✅ All data collection steps completed successfully!")
    else:
        print("⚠️ Some steps failed. Check the output above for details.")
    
    print(f"\n📁 Data saved to: {args.output_dir}")
    print("🔧 Next steps:")
    print("  1. Review the collected data")
    print("  2. Run training with the merged dataset")
    print("  3. Evaluate model performance")

if __name__ == '__main__':
    main()
