import subprocess
import sys
import time
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--interval_minutes', type=int, default=60)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--mode', type=str, default='hot', choices=['hot','new','top'])
    p.add_argument('--output', type=str, default='src/hmama/train/reddit_dataset.jsonl')
    p.add_argument('--image_dir', type=str, default='src/hmama/train/reddit_images')
    p.add_argument('--convert_output', type=str, default='data/reddit_single.jsonl')
    return p.parse_args()


def main():
    args = parse_args()
    while True:
        try:
            subprocess.run([
                sys.executable, 'src/hmama/train/fetch_reddit_dataset.py',
                '--limit', str(args.limit), '--mode', args.mode,
                '--output', args.output, '--image_dir', args.image_dir
            ], check=True)
            subprocess.run([
                sys.executable, 'src/hmama/train/convert_reddit_to_single.py',
                '--input', args.output, '--output', args.convert_output
            ], check=True)
        except Exception as e:
            print('Scheduled run failed:', e)
        time.sleep(args.interval_minutes * 60)


if __name__ == '__main__':
    main()


