import praw
import json
import requests
from PIL import Image
from io import BytesIO
import os
import argparse
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--client_id', type=str, required=False, default=os.getenv('REDDIT_CLIENT_ID', 'gq-ClCo2GSAPCQ4kakFyVg'))
    p.add_argument('--client_secret', type=str, required=False, default=os.getenv('REDDIT_CLIENT_SECRET', 'nRfTP8XorU857YdfNT9GGWAERRr0Aw'))
    p.add_argument('--user_agent', type=str, required=False, default=os.getenv('REDDIT_USER_AGENT', 'python:HMaMaDataFetcher:0.1 (by /u/Maeson_404)'))
    p.add_argument('--subs', type=str, default='news,worldnews,technology,politics,conspiracy,UpliftingNews')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--output', type=str, default='src/hmama/train/reddit_dataset.jsonl')
    p.add_argument('--image_dir', type=str, default='src/hmama/train/reddit_images')
    p.add_argument('--mode', type=str, default='hot', choices=['hot','new','top'])
    p.add_argument('--max_images_per_post', type=int, default=2)
    p.add_argument('--timeout', type=int, default=15)
    return p.parse_args()

def _collect_image_urls(post) -> list:
    urls = []
    # 1) Direct url or overridden dest
    for key in ['url_overridden_by_dest', 'url']:
        u = getattr(post, key, None)
        if isinstance(u, str) and any(u.lower().endswith(ext) for ext in ('jpg','jpeg','png','webp')):
            urls.append(u)
    # 2) Preview images - only get the highest resolution source
    try:
        prev = getattr(post, 'preview', None)
        if prev and 'images' in prev and prev['images']:
            img0 = prev['images'][0]
            if 'source' in img0 and 'url' in img0['source']:
                urls.append(img0['source']['url'].replace('&amp;','&'))
            # Skip resolutions as they're duplicates of the source
    except Exception:
        pass
    # 3) Gallery/media_metadata
    try:
        if getattr(post, 'is_gallery', False):
            media_meta = getattr(post, 'media_metadata', {}) or {}
            for item in media_meta.values():
                s = item.get('s') or {}
                u = s.get('u') or s.get('gif') or s.get('mp4')
                if isinstance(u, str):
                    urls.append(u.replace('&amp;','&'))
    except Exception:
        pass
    # 4) Deduplicate preserve order and remove similar URLs
    seen = set()
    dedup = []
    for u in urls:
        if not u:
            continue
        # Normalize URL for better deduplication
        normalized = u.split('?')[0]  # Remove query parameters
        if normalized not in seen:
            seen.add(normalized)
            dedup.append(u)
    return dedup


def main():
    args = parse_args()
    reddit = praw.Reddit(
        client_id=args.client_id,
        client_secret=args.client_secret,
        user_agent=args.user_agent
    )

    subs = [s.strip() for s in args.subs.split(',') if s.strip()]
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset = []
    stats = {s: {'seen': 0, 'kept': 0} for s in subs}

    for sub in subs:
        if args.mode == 'hot':
            iterator = reddit.subreddit(sub).hot(limit=args.limit)
        elif args.mode == 'new':
            iterator = reddit.subreddit(sub).new(limit=args.limit)
        else:
            iterator = reddit.subreddit(sub).top(limit=args.limit)

        for post in iterator:
            stats[sub]['seen'] += 1
            text = (post.title or '') + " " + (post.selftext or '')
            images = []
            urls = _collect_image_urls(post)[: max(1, args.max_images_per_post)]
            for i, u in enumerate(urls):
                try:
                    resp = requests.get(u, timeout=args.timeout)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                    suffix = '.jpg'
                    try:
                        ext = u.split('?')[0].split('.')[-1].lower()
                        if ext in ['jpg','jpeg','png','webp']:
                            suffix = '.' + ext
                    except Exception:
                        pass
                    img_path = os.path.join(args.image_dir, f"{post.id}_{i}{suffix}")
                    img.save(img_path)
                    images.append(img_path)
                except Exception as e:
                    # keep going
                    continue

            if images:
                stats[sub]['kept'] += 1
                dataset.append({
                    'id': post.id,
                    'created_utc': getattr(post, 'created_utc', None),
                    'subreddit': sub,
                    'text': text.strip(),
                    'ocr': '',
                    'image': images,
                    'label': 0
                })

    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(dataset)} multimodal Reddit posts to {args.output} and images to {args.image_dir}")
    for sub in subs:
        s = stats[sub]
        print(f"{sub}: seen={s['seen']} kept={s['kept']}")


if __name__ == '__main__':
    main()
