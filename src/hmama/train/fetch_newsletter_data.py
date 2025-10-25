import feedparser
import requests
from bs4 import BeautifulSoup
import json
import os
import argparse
from datetime import datetime, timedelta
import time
import hashlib
from urllib.parse import urljoin, urlparse
import re
from PIL import Image
from io import BytesIO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, default='data/raw/newsletter_dataset.jsonl')
    p.add_argument('--image_dir', type=str, default='data/raw/newsletter_images')
    p.add_argument('--days_back', type=int, default=7, help='Number of days back to fetch news')
    p.add_argument('--max_articles_per_source', type=int, default=50)
    p.add_argument('--timeout', type=int, default=15)
    p.add_argument('--delay', type=float, default=1.0, help='Delay between requests to avoid rate limiting')
    return p.parse_args()

# Reliable news sources with RSS feeds
NEWS_SOURCES = {
    'indian': {
        'The Hindu': 'https://www.thehindu.com/news/feeder/default.rss',
        'Times of India': 'https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms',
        'Indian Express': 'https://indianexpress.com/section/india/feed/',
        'Hindustan Times': 'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
        'Mint': 'https://www.livemint.com/rss/news',
        'Business Standard': 'https://www.business-standard.com/rss/home_page_top_stories.cms',
        'NDTV': 'https://feeds.feedburner.com/ndtvnews-india-news',
        'News18': 'https://www.news18.com/rss/india.xml',
        'Republic World': 'https://www.republicworld.com/rss/section/india-news.xml',
        'Zee News': 'https://zeenews.india.com/rss/india.xml'
    },
    'global': {
        'Reuters': 'https://feeds.reuters.com/reuters/topNews',
        'Associated Press': 'https://feeds.apnews.com/rss/apf-topnews',
        'BBC News': 'http://feeds.bbci.co.uk/news/rss.xml',
        'CNN': 'http://rss.cnn.com/rss/edition.rss',
        'The Guardian': 'https://www.theguardian.com/world/rss',
        'New York Times': 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
        'Washington Post': 'http://feeds.washingtonpost.com/rss/world',
        'NPR': 'https://feeds.npr.org/1001/rss.xml',
        'PBS NewsHour': 'https://www.pbs.org/newshour/feeds/rss/world',
        'Al Jazeera': 'https://www.aljazeera.com/xml/rss/all.xml'
    }
}

def extract_images_from_article(url, session, timeout=15):
    """Extract images from article content"""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        images = []
        # Look for images in various common locations
        img_selectors = [
            'img[src]',
            '.article-image img',
            '.content img',
            '.story-content img',
            '.article-body img',
            '.entry-content img'
        ]
        
        for selector in img_selectors:
            for img in soup.select(selector):
                src = img.get('src')
                if src:
                    # Convert relative URLs to absolute
                    src = urljoin(url, src)
                    # Filter out small images, icons, and non-news images
                    if (src and 
                        not any(skip in src.lower() for skip in ['icon', 'logo', 'avatar', 'thumbnail', 'banner']) and
                        any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp'])):
                        images.append(src)
                        if len(images) >= 3:  # Limit to 3 images per article
                            break
            if images:
                break
                
        return images[:3]  # Return max 3 images
    except Exception as e:
        print(f"Error extracting images from {url}: {e}")
        return []

def download_image(url, image_dir, article_id, index, session, timeout=15):
    """Download and save image"""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Validate image
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Generate filename
        ext = url.split('?')[0].split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'webp']:
            ext = 'jpg'
        filename = f"{article_id}_{index}.{ext}"
        filepath = os.path.join(image_dir, filename)
        
        img.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common HTML artifacts
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    
    return text

def is_recent_article(entry, days_back=7):
    """Check if article is within the specified time range"""
    try:
        # Try different date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S %z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        published = entry.get('published_parsed')
        if published:
            pub_date = datetime(*published[:6])
        else:
            # Fallback to current time if no date
            pub_date = datetime.now()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return pub_date >= cutoff_date
    except:
        return True  # Include if date parsing fails

def fetch_newsletter_data():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    dataset = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    stats = {'total_articles': 0, 'with_images': 0, 'sources_processed': 0}
    
    for category, sources in NEWS_SOURCES.items():
        print(f"\nProcessing {category} news sources...")
        
        for source_name, rss_url in sources.items():
            print(f"Fetching from {source_name}...")
            
            try:
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                if feed.bozo:
                    print(f"Warning: RSS feed for {source_name} may have issues")
                
                articles_processed = 0
                
                for entry in feed.entries:
                    if articles_processed >= args.max_articles_per_source:
                        break
                    
                    # Check if article is recent
                    if not is_recent_article(entry, args.days_back):
                        continue
                    
                    # Extract article data
                    title = clean_text(entry.get('title', ''))
                    summary = clean_text(entry.get('summary', ''))
                    link = entry.get('link', '')
                    
                    if not title or not link:
                        continue
                    
                    # Combine title and summary for text content
                    text_content = f"{title}. {summary}".strip()
                    
                    # Extract images from article
                    images = extract_images_from_article(link, session, args.timeout)
                    
                    # Download and save images
                    saved_images = []
                    if images:
                        article_id = hashlib.md5(link.encode()).hexdigest()[:12]
                        
                        for i, img_url in enumerate(images):
                            img_path = download_image(img_url, args.image_dir, article_id, i, session, args.timeout)
                            if img_path:
                                saved_images.append(img_path)
                            
                            # Add delay to avoid rate limiting
                            time.sleep(args.delay)
                    
                    # Create dataset entry
                    entry_data = {
                        'id': hashlib.md5(link.encode()).hexdigest()[:12],
                        'source': source_name,
                        'category': category,
                        'text': text_content,
                        'title': title,
                        'summary': summary,
                        'url': link,
                        'published': entry.get('published', ''),
                        'ocr': '',  # Will be filled during processing
                        'image': saved_images,
                        'label': 1,  # Reliable news sources are labeled as REAL (1)
                        'created_utc': int(datetime.now().timestamp())
                    }
                    
                    dataset.append(entry_data)
                    articles_processed += 1
                    stats['total_articles'] += 1
                    
                    if saved_images:
                        stats['with_images'] += 1
                    
                    # Add delay between articles
                    time.sleep(args.delay)
                
                stats['sources_processed'] += 1
                print(f"Processed {articles_processed} articles from {source_name}")
                
            except Exception as e:
                print(f"Error processing {source_name}: {e}")
                continue
    
    # Save dataset
    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nDataset saved to {args.output}")
    print(f"Total articles: {stats['total_articles']}")
    print(f"Articles with images: {stats['with_images']}")
    print(f"Sources processed: {stats['sources_processed']}")
    print(f"Images saved to: {args.image_dir}")

if __name__ == '__main__':
    fetch_newsletter_data()
