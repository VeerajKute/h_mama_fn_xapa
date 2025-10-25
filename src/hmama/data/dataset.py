import json
from torch.utils.data import Dataset
from PIL import Image
import pytesseract
import os
import random

class NewsItem:
    def __init__(self, d):
        self.id = d['id']
        self.text = d['text']
        self.label = d['label']
        self.source = d.get('source', 'unknown')
        
        # Handle both old format (single image_path) and new format (list of images)
        if 'image_path' in d:
            # Old format - single image
            self.images = [d['image_path']] if d['image_path'] else []
        elif 'image' in d:
            # New format - list of images
            self.images = d['image'] if isinstance(d['image'], list) else [d['image']] if d['image'] else []
        else:
            self.images = []
        
        # Additional metadata
        self.ocr = d.get('ocr', '')
        self.url = d.get('url', '')
        self.created_utc = d.get('created_utc', 0)

class NewsDataset(Dataset):
    def __init__(self, jsonl_path, max_images_per_sample=3, use_ocr=True):
        self.items = []
        self.max_images_per_sample = max_images_per_sample
        self.use_ocr = use_ocr
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.items.append(NewsItem(json.loads(line)))
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Skipping invalid line: {e}")
                        continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        
        # Load and process images
        images = []
        ocr_texts = []
        
        # Limit number of images per sample
        selected_images = it.images[:self.max_images_per_sample]
        
        for img_path in selected_images:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    
                    if self.use_ocr:
                        ocr_text = pytesseract.image_to_string(img)[:400]
                        ocr_texts.append(ocr_text)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue
        
        # If no images were loaded, create a placeholder
        if not images:
            # Create a blank image as placeholder
            placeholder = Image.new('RGB', (224, 224), color='white')
            images = [placeholder]
            if self.use_ocr:
                ocr_texts = ['']
        
        # Combine OCR texts
        combined_ocr = ' '.join(ocr_texts) if ocr_texts else it.ocr
        
        return {
            'id': it.id,
            'text': it.text,
            'ocr': combined_ocr,
            'image': images[0] if len(images) == 1 else images,  # Return single image or list
            'images': images,  # Always return list for compatibility
            'label': it.label,
            'source': it.source,
            'url': it.url,
            'created_utc': it.created_utc
        }

class MultiModalNewsDataset(Dataset):
    """Enhanced dataset for multimodal fake news detection with multiple data sources"""
    
    def __init__(self, jsonl_path, max_images_per_sample=3, use_ocr=True, 
                 balance_sources=True, min_samples_per_source=10):
        self.items = []
        self.max_images_per_sample = max_images_per_sample
        self.use_ocr = use_ocr
        self.balance_sources = balance_sources
        self.min_samples_per_source = min_samples_per_source
        
        # Load data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.items.append(NewsItem(json.loads(line)))
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Skipping invalid line: {e}")
                        continue
        
        # Balance sources if requested
        if self.balance_sources:
            self._balance_sources()
    
    def _balance_sources(self):
        """Balance samples across different sources"""
        source_counts = {}
        for item in self.items:
            source = item.source
            if source not in source_counts:
                source_counts[source] = []
            source_counts[source].append(item)
        
        # Find minimum count across sources
        min_count = min(len(samples) for samples in source_counts.values())
        min_count = max(min_count, self.min_samples_per_source)
        
        # Sample from each source
        balanced_items = []
        for source, samples in source_counts.items():
            if len(samples) >= min_count:
                selected = random.sample(samples, min_count)
                balanced_items.extend(selected)
            else:
                balanced_items.extend(samples)
        
        self.items = balanced_items
        print(f"Balanced dataset: {len(self.items)} samples from {len(source_counts)} sources")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        
        # Load and process images
        images = []
        ocr_texts = []
        
        # Limit number of images per sample
        selected_images = it.images[:self.max_images_per_sample]
        
        for img_path in selected_images:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    
                    if self.use_ocr:
                        ocr_text = pytesseract.image_to_string(img)[:400]
                        ocr_texts.append(ocr_text)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue
        
        # If no images were loaded, create a placeholder
        if not images:
            # Create a blank image as placeholder
            placeholder = Image.new('RGB', (224, 224), color='white')
            images = [placeholder]
            if self.use_ocr:
                ocr_texts = ['']
        
        # Combine OCR texts
        combined_ocr = ' '.join(ocr_texts) if ocr_texts else it.ocr
        
        return {
            'id': it.id,
            'text': it.text,
            'ocr': combined_ocr,
            'image': images[0] if len(images) == 1 else images,  # Return single image or list
            'images': images,  # Always return list for compatibility
            'label': it.label,
            'source': it.source,
            'url': it.url,
            'created_utc': it.created_utc
        }
