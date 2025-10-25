import os
import json
import pandas as pd
from pathlib import Path

root = Path("data/raw/FakeNewsNet/GossipCop")  # or PolitiFact
out_file = Path("data/raw/fakenewsnet.jsonl")
images_dir = root / "images"

with open(out_file, "w") as f_out:
    for news_id in os.listdir(root / "news content"):
        try:
            text_file = root / "news content" / news_id / "news content.json"
            data = json.load(open(text_file))
            text = data.get("text", "")
            label = 1 if data.get("label")=="fake" else 0
            image_path = images_dir / f"{news_id}.jpg"
            if not image_path.exists():
                image_path = None
            entry = {
                "id": news_id,
                "text": text,
                "image_path": str(image_path) if image_path else "",
                "label": label,
                "timestamp": data.get("publish_date", ""),
                "source": data.get("source_url","")
            }
            f_out.write(json.dumps(entry)+"\n")
        except Exception as e:
            print(f"Skipping {news_id}: {e}")
