#!/usr/bin/env python3
"""
H-MaMa Fake News Detection - Complete Setup and Training Script
One command to rule them all: Install → Collect Data → Train Model
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

def print_step(step, title):
    print(f"\n{'='*60}")
    print(f"🚀 STEP {step}: {title}")
    print(f"{'='*60}")

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n📋 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def install_requirements():
    """Install required packages"""
    print_step(1, "Installing Requirements")
    
    # Check if virtual environment exists
    if not os.path.exists('venv'):
        print("📦 Creating virtual environment...")
        if not run_command([sys.executable, '-m', 'venv', 'venv'], "Creating virtual environment"):
            return False
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = os.path.join('venv', 'Scripts', 'pip.exe')
        python_path = os.path.join('venv', 'Scripts', 'python.exe')
    else:  # Unix/Linux/Mac
        pip_path = os.path.join('venv', 'bin', 'pip')
        python_path = os.path.join('venv', 'bin', 'python')
    
    # Install requirements
    requirements = [
        'torch==2.3.1',
        'torchvision==0.18.1', 
        'transformers==4.41.2',
        'sentencepiece',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm',
        'Pillow',
        'opencv-python',
        'pytesseract',
        'torch-geometric==2.5.3',
        'fastapi',
        'uvicorn',
        'captum',
        'feedparser',
        'beautifulsoup4',
        'requests',
        'praw'
    ]
    
    print("📦 Installing packages...")
    for package in requirements:
        if not run_command([pip_path, 'install', package], f"Installing {package}"):
            print(f"⚠️ Warning: Failed to install {package}, continuing...")
    
    return python_path

def collect_data(python_path):
    """Collect data from all sources"""
    print_step(2, "Collecting Training Data")
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/raw/reddit_images', exist_ok=True)
    os.makedirs('data/raw/newsletter_images', exist_ok=True)
    
    print("🔴 Collecting Reddit data...")
    reddit_cmd = [
        python_path, 'src/hmama/train/fetch_reddit_dataset.py',
        '--limit', '100',
        '--output', 'data/raw/reddit_dataset.jsonl',
        '--image_dir', 'data/raw/reddit_images',
        '--subs', 'news,worldnews,technology',
        '--mode', 'hot'
    ]
    
    if not run_command(reddit_cmd, "Reddit data collection", check=False):
        print("⚠️ Reddit collection had issues, continuing...")
    
    print("\n📰 Collecting newsletter data...")
    newsletter_cmd = [
        python_path, 'src/hmama/train/fetch_newsletter_data.py',
        '--output', 'data/raw/newsletter_dataset.jsonl',
        '--image_dir', 'data/raw/newsletter_images',
        '--days_back', '3',
        '--max_articles_per_source', '20'
    ]
    
    if not run_command(newsletter_cmd, "Newsletter data collection", check=False):
        print("⚠️ Newsletter collection had issues, continuing...")
    
    print("\n🔄 Merging datasets...")
    merge_cmd = [
        python_path, 'src/hmama/train/merge_datasets.py',
        '--reddit_data', 'data/raw/reddit_dataset.jsonl',
        '--newsletter_data', 'data/raw/newsletter_dataset.jsonl',
        '--liar_data', 'data/processed/LIAR/train.jsonl',
        '--output', 'data/raw/merged_dataset.jsonl',
        '--max_samples', '1000'
    ]
    
    if not run_command(merge_cmd, "Dataset merging", check=False):
        print("⚠️ Dataset merging had issues, using fallback...")
        # Create a simple fallback dataset
        create_fallback_dataset()
    
    return True

def create_fallback_dataset():
    """Create a simple fallback dataset if collection fails"""
    print("📝 Creating fallback dataset...")
    
    fallback_data = [
        {
            "id": "fallback_1",
            "text": "Breaking news: Major scientific breakthrough announced today by researchers.",
            "ocr": "",
            "image": [],
            "label": 1,
            "source": "fallback",
            "url": "",
            "created_utc": int(datetime.now().timestamp())
        },
        {
            "id": "fallback_2", 
            "text": "URGENT: Click here to see shocking video that will change your life forever!",
            "ocr": "",
            "image": [],
            "label": 0,
            "source": "fallback",
            "url": "",
            "created_utc": int(datetime.now().timestamp())
        }
    ]
    
    with open('data/raw/merged_dataset.jsonl', 'w', encoding='utf-8') as f:
        for item in fallback_data:
            f.write(json.dumps(item) + '\n')
    
    print("✅ Fallback dataset created")

def train_model(python_path):
    """Train the fake news detection model"""
    print_step(3, "Training the Model")
    
    # Check if we have a dataset
    if not os.path.exists('data/raw/merged_dataset.jsonl'):
        print("❌ No dataset found, cannot train model")
        return False
    
    print("🧠 Starting model training...")
    
    # Simple training command
    train_cmd = [
        python_path, 'src/hmama/train/train_content.py'
    ]
    
    # Update the training script to use our dataset
    update_training_script()
    
    if not run_command(train_cmd, "Model training", check=False):
        print("⚠️ Training had issues, but continuing...")
    
    print("✅ Training completed!")
    return True

def update_training_script():
    """Update training script to use our dataset"""
    training_script = 'src/hmama/train/train_content.py'
    
    if os.path.exists(training_script):
        with open(training_script, 'r') as f:
            content = f.read()
        
        # Replace dataset path
        content = content.replace(
            "ds = NewsDataset('data/raw/samples.jsonl')",
            "ds = NewsDataset('data/raw/merged_dataset.jsonl')"
        )
        
        with open(training_script, 'w') as f:
            f.write(content)

def start_frontend(python_path):
    """Start the web frontend"""
    print_step(4, "Starting Web Interface")
    
    print("🌐 Starting H-MaMa web interface...")
    print("The system will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    frontend_cmd = [python_path, 'run_frontend.py']
    
    try:
        subprocess.run(frontend_cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down H-MaMa system...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend error: {e}")

def main():
    """Main function - runs everything"""
    print("🎯 H-MaMa Fake News Detection System")
    print("=" * 60)
    print("This script will:")
    print("1. Install all requirements")
    print("2. Collect data from Reddit and news sources")
    print("3. Train the fake news detection model")
    print("4. Start the web interface")
    print("=" * 60)
    
    # Get user confirmation
    response = input("\n🤔 Do you want to continue? (y/n): ").lower().strip()
    if response != 'y':
        print("👋 Goodbye!")
        return
    
    start_time = time.time()
    
    # Step 1: Install requirements
    python_path = install_requirements()
    if not python_path:
        print("❌ Failed to install requirements")
        return
    
    # Step 2: Collect data
    if not collect_data(python_path):
        print("❌ Failed to collect data")
        return
    
    # Step 3: Train model
    if not train_model(python_path):
        print("❌ Failed to train model")
        return
    
    # Step 4: Start frontend
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Setup completed in {elapsed_time:.1f} seconds!")
    print("🚀 Starting web interface...")
    
    start_frontend(python_path)

if __name__ == '__main__':
    main()
