#!/usr/bin/env python3
"""
Data validation script for H-MaMa fake news detection system.
Validates data quality, checks for duplicates, and provides statistics.
"""

import json
import os
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from PIL import Image
import re

def parse_args():
    p = argparse.ArgumentParser(description='Validate collected data quality')
    p.add_argument('--dataset', type=str, required=True, help='Path to dataset JSONL file')
    p.add_argument('--output', type=str, default='data_validation_report.json', help='Output report file')
    p.add_argument('--check_images', action='store_true', help='Check image file integrity')
    p.add_argument('--check_duplicates', action='store_true', help='Check for duplicate content')
    p.add_argument('--min_text_length', type=int, default=10, help='Minimum text length')
    p.add_argument('--max_text_length', type=int, default=10000, help='Maximum text length')
    return p.parse_args()

def load_dataset(file_path):
    """Load dataset from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    item['_line_number'] = line_num
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
    return data

def validate_text_quality(data, min_length=10, max_length=10000):
    """Validate text quality metrics"""
    issues = []
    stats = {
        'total_texts': len(data),
        'empty_texts': 0,
        'too_short': 0,
        'too_long': 0,
        'avg_length': 0,
        'text_lengths': []
    }
    
    total_length = 0
    
    for item in data:
        text = item.get('text', '')
        text_length = len(text.strip())
        stats['text_lengths'].append(text_length)
        total_length += text_length
        
        if not text.strip():
            stats['empty_texts'] += 1
            issues.append(f"Empty text in item {item.get('id', 'unknown')}")
        elif text_length < min_length:
            stats['too_short'] += 1
            issues.append(f"Text too short ({text_length} chars) in item {item.get('id', 'unknown')}")
        elif text_length > max_length:
            stats['too_long'] += 1
            issues.append(f"Text too long ({text_length} chars) in item {item.get('id', 'unknown')}")
    
    stats['avg_length'] = total_length / len(data) if data else 0
    
    return stats, issues

def validate_images(data, check_integrity=False):
    """Validate image data"""
    issues = []
    stats = {
        'total_items': len(data),
        'items_with_images': 0,
        'total_images': 0,
        'missing_images': 0,
        'corrupt_images': 0,
        'image_formats': Counter(),
        'image_sizes': []
    }
    
    for item in data:
        images = item.get('image', [])
        if not isinstance(images, list):
            images = [images] if images else []
        
        if images:
            stats['items_with_images'] += 1
            stats['total_images'] += len(images)
            
            for img_path in images:
                if not os.path.exists(img_path):
                    stats['missing_images'] += 1
                    issues.append(f"Missing image file: {img_path}")
                elif check_integrity:
                    try:
                        with Image.open(img_path) as img:
                            stats['image_formats'][img.format] += 1
                            stats['image_sizes'].append(img.size)
                    except Exception as e:
                        stats['corrupt_images'] += 1
                        issues.append(f"Corrupt image {img_path}: {e}")
        else:
            stats['missing_images'] += 1
    
    return stats, issues

def check_duplicates(data):
    """Check for duplicate content"""
    issues = []
    stats = {
        'total_items': len(data),
        'duplicate_texts': 0,
        'duplicate_urls': 0,
        'duplicate_ids': 0
    }
    
    # Check for duplicate texts
    text_hashes = {}
    url_set = set()
    id_set = set()
    
    for item in data:
        text = item.get('text', '').strip()
        url = item.get('url', '')
        item_id = item.get('id', '')
        
        # Check duplicate text
        if text:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in text_hashes:
                stats['duplicate_texts'] += 1
                issues.append(f"Duplicate text: {item.get('id', 'unknown')} matches {text_hashes[text_hash]}")
            else:
                text_hashes[text_hash] = item.get('id', 'unknown')
        
        # Check duplicate URLs
        if url:
            if url in url_set:
                stats['duplicate_urls'] += 1
                issues.append(f"Duplicate URL: {url}")
            else:
                url_set.add(url)
        
        # Check duplicate IDs
        if item_id:
            if item_id in id_set:
                stats['duplicate_ids'] += 1
                issues.append(f"Duplicate ID: {item_id}")
            else:
                id_set.add(item_id)
    
    return stats, issues

def analyze_sources(data):
    """Analyze data sources and distribution"""
    stats = {
        'total_items': len(data),
        'sources': Counter(),
        'labels': Counter(),
        'source_label_distribution': defaultdict(Counter)
    }
    
    for item in data:
        source = item.get('source', 'unknown')
        label = item.get('label', 'unknown')
        
        stats['sources'][source] += 1
        stats['labels'][label] += 1
        stats['source_label_distribution'][source][label] += 1
    
    return stats

def validate_data_format(data):
    """Validate data format and required fields"""
    issues = []
    stats = {
        'total_items': len(data),
        'missing_required_fields': 0,
        'invalid_labels': 0,
        'invalid_sources': 0
    }
    
    required_fields = ['id', 'text', 'label']
    valid_labels = [0, 1, '0', '1']
    valid_sources = ['reddit', 'newsletter', 'liar', 'unknown']
    
    for item in data:
        # Check required fields
        missing_fields = [field for field in required_fields if field not in item or not item[field]]
        if missing_fields:
            stats['missing_required_fields'] += 1
            issues.append(f"Missing required fields {missing_fields} in item {item.get('id', 'unknown')}")
        
        # Check label validity
        label = item.get('label')
        if label not in valid_labels:
            stats['invalid_labels'] += 1
            issues.append(f"Invalid label {label} in item {item.get('id', 'unknown')}")
        
        # Check source validity
        source = item.get('source', 'unknown')
        if source not in valid_sources:
            stats['invalid_sources'] += 1
            issues.append(f"Invalid source {source} in item {item.get('id', 'unknown')}")
    
    return stats, issues

def generate_report(data, args):
    """Generate comprehensive validation report"""
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'dataset_file': args.dataset,
        'total_items': len(data),
        'validation_settings': {
            'check_images': args.check_images,
            'check_duplicates': args.check_duplicates,
            'min_text_length': args.min_text_length,
            'max_text_length': args.max_text_length
        }
    }
    
    all_issues = []
    
    # Text quality validation
    print("Validating text quality...")
    text_stats, text_issues = validate_text_quality(data, args.min_text_length, args.max_text_length)
    report['text_quality'] = text_stats
    all_issues.extend(text_issues)
    
    # Image validation
    print("Validating images...")
    image_stats, image_issues = validate_images(data, args.check_images)
    report['image_quality'] = image_stats
    all_issues.extend(image_issues)
    
    # Duplicate checking
    if args.check_duplicates:
        print("Checking for duplicates...")
        duplicate_stats, duplicate_issues = check_duplicates(data)
        report['duplicates'] = duplicate_stats
        all_issues.extend(duplicate_issues)
    
    # Source analysis
    print("Analyzing sources...")
    source_stats = analyze_sources(data)
    report['source_analysis'] = source_stats
    
    # Format validation
    print("Validating data format...")
    format_stats, format_issues = validate_data_format(data)
    report['format_validation'] = format_stats
    all_issues.extend(format_issues)
    
    # Summary
    report['summary'] = {
        'total_issues': len(all_issues),
        'critical_issues': len([i for i in all_issues if 'Missing' in i or 'Corrupt' in i]),
        'warnings': len([i for i in all_issues if 'too short' in i or 'too long' in i]),
        'data_quality_score': max(0, 100 - len(all_issues) * 2)  # Simple scoring
    }
    
    report['issues'] = all_issues
    
    return report

def main():
    args = parse_args()
    
    print(f"🔍 Validating dataset: {args.dataset}")
    
    # Load data
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} items")
    
    # Generate report
    report = generate_report(data, args)
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n📊 Validation Report Summary:")
    print(f"Total items: {report['total_items']}")
    print(f"Total issues: {report['summary']['total_issues']}")
    print(f"Critical issues: {report['summary']['critical_issues']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Data quality score: {report['summary']['data_quality_score']}/100")
    
    # Print source distribution
    print(f"\n📈 Source Distribution:")
    for source, count in report['source_analysis']['sources'].most_common():
        print(f"  {source}: {count}")
    
    # Print label distribution
    print(f"\n🏷️ Label Distribution:")
    for label, count in report['source_analysis']['labels'].most_common():
        print(f"  {label}: {count}")
    
    # Print text quality
    print(f"\n📝 Text Quality:")
    print(f"  Average length: {report['text_quality']['avg_length']:.1f} characters")
    print(f"  Empty texts: {report['text_quality']['empty_texts']}")
    print(f"  Too short: {report['text_quality']['too_short']}")
    print(f"  Too long: {report['text_quality']['too_long']}")
    
    # Print image quality
    print(f"\n🖼️ Image Quality:")
    print(f"  Items with images: {report['image_quality']['items_with_images']}")
    print(f"  Total images: {report['image_quality']['total_images']}")
    print(f"  Missing images: {report['image_quality']['missing_images']}")
    if args.check_images:
        print(f"  Corrupt images: {report['image_quality']['corrupt_images']}")
    
    print(f"\n📋 Full report saved to: {args.output}")
    
    if report['summary']['total_issues'] > 0:
        print(f"\n⚠️ Issues found:")
        for issue in report['issues'][:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(report['issues']) > 10:
            print(f"  ... and {len(report['issues']) - 10} more issues")

if __name__ == '__main__':
    main()
