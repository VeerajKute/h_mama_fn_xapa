import os
import json
from typing import Optional, List
from torch.utils.data import Dataset
from PIL import Image
import pytesseract
import torch

# -----------------------------
# Original dataset class
# -----------------------------
class LocalMultimodalDataset(Dataset):
    """
    Load multimodal data from a directory with a single JSONL file listing entries.
    Each line should be a JSON object with:
      - id: unique id
      - text: claim/text
      - image_path: relative path to image file
      - label: 0 (fake) or 1 (real)
      - timestamp, source (optional)
    """
    LABEL_MAP = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 1,
        "mostly-true": 1,
        "true": 1
    }

    def __init__(self, jsonl_path: str, image_root: Optional[str] = None, max_ocr_chars: int = 400):
        assert os.path.exists(jsonl_path), f"File not found: {jsonl_path}"
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.image_root = image_root or os.path.dirname(jsonl_path)
        self.max_ocr_chars = max_ocr_chars

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> Image.Image:
        # 1) Absolute path
        if os.path.isabs(path) and os.path.exists(path):
            return Image.open(path).convert('RGB')
        # 2) Relative to image_root (same dir as JSONL by default)
        p1 = os.path.join(self.image_root, path)
        if os.path.exists(p1):
            return Image.open(p1).convert('RGB')
        # 3) Relative to current working directory (project root when running)
        p2 = os.path.join(os.getcwd(), path)
        if os.path.exists(p2):
            return Image.open(p2).convert('RGB')
        # 4) As-is if exists after normpath
        p3 = os.path.normpath(path)
        if os.path.exists(p3):
            return Image.open(p3).convert('RGB')
        raise FileNotFoundError(f"Image not found: {p1}")

    def __getitem__(self, idx):
        it = self.items[idx]
        img = None
        if 'image_path' in it and it['image_path']:
            img = self._load_image(it['image_path'])
        ocr_text = ''
        if img is not None:
            try:
                ocr_text = pytesseract.image_to_string(img)[:self.max_ocr_chars]
            except Exception:
                ocr_text = ''

        raw_label = str(it.get('label', '')).lower()
        label = self.LABEL_MAP.get(raw_label, 0)

        return {
            'id': it.get('id', str(idx)),
            'text': it.get('text', ''),
            'ocr': ocr_text,
            'image': img,
            'label': label,
            'meta': {k:v for k,v in it.items() if k not in ['text','image_path','label']}
        }

# -----------------------------
# Reddit dataset class
# -----------------------------
class LocalMultimodalDatasetWithImages(Dataset):
    """
    Supports Reddit dataset where each post can have multiple images.
    JSONL format example:
    {"text": "...", "ocr": "", "image": ["reddit_images/1.jpg", "reddit_images/2.jpg"], "label": 0}
    """
    def __init__(self, jsonl_path: str, image_root: Optional[str] = None, max_ocr_chars: int = 400):
        assert os.path.exists(jsonl_path), f"File not found: {jsonl_path}"
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.image_root = image_root or os.path.dirname(jsonl_path)
        self.max_ocr_chars = max_ocr_chars

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> Image.Image:
        p = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert('RGB')

    def __getitem__(self, idx):
        it = self.items[idx]
        images: List[Image.Image] = []
        ocr_texts: List[str] = []

        if 'image' in it and it['image']:
            for img_path in it['image']:
                try:
                    img = self._load_image(img_path)
                    images.append(img)
                    try:
                        ocr_text = pytesseract.image_to_string(img)[:self.max_ocr_chars]
                    except Exception:
                        ocr_text = ''
                    ocr_texts.append(ocr_text)
                except FileNotFoundError:
                    continue  # skip missing images

        combined_ocr = " ".join(ocr_texts)

        label = it.get('label', 0)
        if isinstance(label, str):
            try:
                label = int(label)
            except:
                label = 0

        return {
            'id': it.get('id', str(idx)),
            'text': it.get('text', ''),
            'ocr': combined_ocr,
            'image': images,  # list of PIL images
            'label': label,
            'meta': {k:v for k,v in it.items() if k not in ['text','image','label']}
        }

# -----------------------------
# Collate function for Reddit
# -----------------------------

def reddit_collate_fn(batch):
    """
    Collate function to handle multiple images per sample.
    Converts numerical fields to tensors.
    """
    collated = {
        'id': [],
        'text': [],
        'ocr': [],
        'image': [],
        'label': [],
        'meta': []
    }

    for item in batch:
        collated['id'].append(item['id'])
        collated['text'].append(item['text'])
        collated['ocr'].append(item['ocr'])
        collated['image'].append(item['image'])  # still PIL images
        collated['label'].append(item['label'])
        collated['meta'].append(item['meta'])

    # Convert labels to tensor
    collated['label'] = torch.tensor(collated['label'], dtype=torch.float32)

    return collated
