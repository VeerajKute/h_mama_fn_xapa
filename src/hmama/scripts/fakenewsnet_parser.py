import os
import json
from pathlib import Path

def parse_fakenewsnet(root_dir, out_jsonl):
    """
    Simple parser for FakeNewsNet-style directory.
    This expects the dataset has folders for claims/articles and associated images.
    The parser is conservative: it will look for files containing 'content' or 'article'
    and images in 'images' subfolders. You should adapt it to the exact dataset layout.

    Usage:
      parse_fakenewsnet('/path/to/FakeNewsNet/GossipCop', 'data/raw/fakenewsnet_gossipcop.jsonl')
    """
    root = Path(root_dir)
    entries = []
    # naive search: all .json or .txt files
    for p in root.rglob('*'):
        if p.suffix.lower() in ['.json', '.txt']:
            # try to read as json; fallback to text
            try:
                data = json.loads(p.read_text())
                text = data.get('content') or data.get('text') or data.get('article') or ''
            except Exception:
                text = p.read_text()
            # find nearby images
            img = None
            img_dir = p.parent / 'images'
            if img_dir.exists():
                imgs = list(img_dir.glob('*'))
                if imgs:
                    img = str(imgs[0])
            # label unknown: set to 0 (user must label or merge with fact-check files)
            entries.append({'id': str(p), 'text': text, 'image_path': img or '', 'label': 0})
    # write out
    with open(out_jsonl, 'w') as fo:
        for e in entries:
            fo.write(json.dumps(e) + '\n')
    print(f'Parsed {len(entries)} items to {out_jsonl}')
