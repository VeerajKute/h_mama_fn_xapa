import json
import csv
from pathlib import Path

def convert_tsv_to_jsonl(tsv_path, jsonl_path):
    with open(tsv_path, 'r', encoding='utf-8') as f_in, open(jsonl_path, 'w', encoding='utf-8') as f_out:
        reader = csv.reader(f_in, delimiter='\t')
        for row in reader:
            label = row[1]  # "pants-fire", "false", "true", etc.
            text = row[2]   # statement
            obj = {
                "id": row[0],
                "text": text,
                "label": label,
                "image_path": None,   # No images in LIAR
                "timestamp": None,
                "source": None
            }
            f_out.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
    root = Path("data/raw/LIAR")
    out = Path("data/processed/LIAR")
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test", "valid"]:
        convert_tsv_to_jsonl(root / f"{split}.tsv", out / f"{split}.jsonl")
