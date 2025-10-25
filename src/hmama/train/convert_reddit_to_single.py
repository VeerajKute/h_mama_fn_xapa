import json
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='src/hmama/train/reddit_dataset.jsonl')
    p.add_argument('--output', type=str, default='data/reddit_single.jsonl')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    count = 0
    with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            obj = json.loads(line)
            text = obj.get('text', '')
            label = obj.get('label', 0)
            images = obj.get('image', [])
            img_path = images[0] if isinstance(images, list) and images else None
            if not img_path:
                continue
            # Resolve absolute path robustly (handles already-rooted relative paths)
            if os.path.isabs(img_path):
                abs_img = os.path.normpath(img_path)
            else:
                cand1 = os.path.normpath(os.path.join(os.getcwd(), img_path))
                if os.path.exists(cand1):
                    abs_img = cand1
                else:
                    cand2 = os.path.normpath(os.path.join(os.path.dirname(args.input), img_path))
                    abs_img = cand2
            # Rebase to be relative to the output JSONL directory
            rel_img = os.path.relpath(abs_img, start=os.path.dirname(args.output))
            rec = {
                'id': obj.get('id', str(count)),
                'text': text,
                'image_path': rel_img,
                'label': label
            }
            fout.write(json.dumps(rec) + '\n')
            count += 1
    print(f'Wrote {count} records to {args.output}')


if __name__ == '__main__':
    main()


