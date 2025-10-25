#!/usr/bin/env python3
"""Universal Multimodal + Propagation Dataset Converter

Attempts to convert common fake-news dataset folder layouts (FakeNewsNet, Weibo, Twitter-*)
into a unified JSONL for multimodal samples and a propagation JSONL for cascades.

Usage:
  python universal_parser.py --root /path/to/dataset --out_jsonl out.jsonl --out_prop out_prop.jsonl

Notes:
 - This is a heuristic converter. Datasets vary; after conversion inspect the outputs.
 - It looks for:
    * JSON or TXT files containing article/content text
    * images in 'images' subfolders
    * tweet/post JSON files containing 'user', 'retweet', 'created_at' fields
 - For Weibo or other datasets, adapt mapping functions at the bottom of the script.
"""

import argparse, os, json, re
from pathlib import Path
from datetime import datetime

def find_text_files(root):
    exts = ['.json', '.txt', '.html']
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in exts:
            yield p

def read_text_from_file(p):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        # possible keys
        for k in ('content','text','article','title','full_text'):
            if k in data:
                return data.get(k) or data.get('content', '') or ''
        # fallback: stringify certain fields
        if 'tweet' in data:
            return data.get('tweet','')
        # else join all string fields
        s = []
        for v in data.values():
            if isinstance(v,str):
                s.append(v)
        return ' '.join(s)[:20000]
    except Exception:
        try:
            return p.read_text(encoding='utf-8')
        except Exception:
            return ''

def find_images_near(p):
    # look for images in same folder or parent 'images' dirs
    parent = p.parent
    imgs = []
    for cand in list(parent.glob('*')) + list((parent/'images').glob('*')) if (parent/'images').exists() else list(parent.glob('*')):
        if cand.suffix.lower() in ['.jpg','.jpeg','.png','.webp']:
            imgs.append(str(cand))
    return imgs

def parse_fake_news_net(root, out_jsonl, out_prop):
    entries = []
    prop_events = []
    idx=0
    for p in find_text_files(root):
        txt = read_text_from_file(p)
        imgs = find_images_near(p)
        label = 0
        # attempt to infer label from parent folder name (fake/real)
        parts = [pp.lower() for pp in p.parts]
        if any('fake' in pp for pp in parts):
            label = 0
        if any('real' in pp for pp in parts):
            label = 1
        entries.append({'id': f'fnn_{idx}', 'text': txt, 'image_path': imgs[0] if imgs else '', 'label': label})
        # try to build propagation from possible tweets folder
        tweet_dir = p.parent / 'tweets'
        if tweet_dir.exists():
            for tfile in tweet_dir.glob('*.json'):
                try:
                    t = json.loads(tfile.read_text(encoding='utf-8'))
                    user = t.get('user', {}).get('id_str') or t.get('user', {}).get('id') or t.get('user_id') or t.get('userid')
                    created = t.get('created_at') or t.get('timestamp') or t.get('time')
                    if created:
                        try:
                            ts = int(created)
                        except:
                            try:
                                ts = int(datetime.fromisoformat(created).timestamp())
                            except:
                                ts = None
                    else:
                        ts = None
                    prop_events.append({'post_id': f'fnn_{idx}', 'user_id': str(user or 'unk'), 'action': 'share', 'time': ts or 0, 'source': t.get('source','')})
                except Exception:
                    continue
        idx+=1
    # write outputs
    with open(out_jsonl,'w',encoding='utf-8') as fo:
        for e in entries:
            fo.write(json.dumps(e,ensure_ascii=False)+'\n')
    with open(out_prop,'w',encoding='utf-8') as fo:
        for ev in prop_events:
            fo.write(json.dumps(ev,ensure_ascii=False)+'\n')
    print(f'Wrote {len(entries)} samples and {len(prop_events)} propagation events.')

def generic_post_parser(root, out_jsonl, out_prop):
    # Looks for social posts (tweet-like) and groups by post id
    posts = {}
    prop_events = []
    for p in Path(root).rglob('*.json'):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except:
            continue
        # detect tweet-like
        if isinstance(data, dict):
            if 'text' in data and ('user' in data or 'user_id' in data):
                post_id = data.get('id_str') or data.get('id') or (p.stem)
                text = data.get('text') or data.get('full_text') or ''
                imgs = []
                media = data.get('entities',{}).get('media',[]) or data.get('extended_entities',{}).get('media',[])
                for m in media or []:
                    url = m.get('media_url') or m.get('media_url_https') or m.get('url')
                    if url:
                        imgs.append(url)
                label = data.get('label') or 0
                posts[post_id] = {'id':post_id,'text':text,'image_path': imgs[0] if imgs else '', 'label': label}
                # propagation
                user = data.get('user',{}).get('id_str') or data.get('user_id') or data.get('user',{}).get('id')
                rt = data.get('retweeted_status') or data.get('in_reply_to_status_id') or data.get('retweet_id')
                ts = data.get('timestamp_ms') or data.get('created_at')
                try:
                    time_val = int(ts) if isinstance(ts,(int,str)) and str(ts).isdigit() else 0
                except:
                    time_val = 0
                prop_events.append({'post_id': post_id, 'user_id': str(user or 'u0'), 'action':'post' if not rt else 'share', 'time': time_val, 'source': data.get('source','')})
    # write
    with open(out_jsonl,'w',encoding='utf-8') as fo:
        for v in posts.values():
            fo.write(json.dumps(v, ensure_ascii=False) + '\n')
    with open(out_prop,'w',encoding='utf-8') as fo:
        for ev in prop_events:
            fo.write(json.dumps(ev, ensure_ascii=False) + '\n')
    print(f'Wrote {len(posts)} posts and {len(prop_events)} propagation events.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--out_jsonl', type=str, default='data/raw/converted.jsonl')
    parser.add_argument('--out_prop', type=str, default='data/raw/converted_propagation.jsonl')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto','fakenewsnet','social_posts'])
    args = parser.parse_args()
    root = Path(args.root)
    if args.mode == 'fakenewsnet':
        parse_fake_news_net(root, args.out_jsonl, args.out_prop)
    elif args.mode == 'social_posts':
        generic_post_parser(root, args.out_jsonl, args.out_prop)
    else:
        # auto: choose based on heuristics
        # if directory contains subfolders named 'fake' or 'real' -> fakenewsnet
        names = [p.name.lower() for p in root.iterdir() if p.is_dir()]
        if any('fake' in n or 'real' in n for n in names):
            parse_fake_news_net(root, args.out_jsonl, args.out_prop)
        else:
            generic_post_parser(root, args.out_jsonl, args.out_prop)

if __name__ == '__main__':
    main()
