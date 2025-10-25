# 📊 H-MaMa Data Collection System

This document describes the comprehensive data collection system for the H-MaMa fake news detection platform, which now supports multiple data sources including Reddit, reliable newsletters, and existing datasets.

## 🎯 Overview

The H-MaMa system now collects data from multiple sources to create a balanced, diverse training dataset:

- **Reddit Data**: Social media posts with images from various subreddits
- **Newsletter Data**: Reliable news articles from Indian and global sources via RSS feeds
- **LIAR Dataset**: Existing labeled dataset for fake news detection
- **Merged Dataset**: Balanced combination of all sources

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect All Data (Recommended)

```bash
python src/hmama/train/collect_all_data.py --reddit_limit 200 --newsletter_days 7
```

### 3. Test Data Collection

```bash
python test_data_collection.py
```

## 📁 Data Sources

### Reddit Data Collection

**Script**: `src/hmama/train/fetch_reddit_dataset.py`

**Features**:
- Collects posts from multiple subreddits (news, worldnews, technology, politics, etc.)
- Downloads associated images
- Prevents duplicate image downloads
- Supports different sorting modes (hot, new, top)

**Usage**:
```bash
python src/hmama/train/fetch_reddit_dataset.py \
    --limit 200 \
    --subs "news,worldnews,technology" \
    --output data/raw/reddit_dataset.jsonl \
    --image_dir data/raw/reddit_images
```

**Parameters**:
- `--limit`: Number of posts to collect per subreddit
- `--subs`: Comma-separated list of subreddits
- `--mode`: Sorting mode (hot, new, top)
- `--max_images_per_post`: Maximum images per post (default: 2)
- `--timeout`: Request timeout in seconds

### Newsletter Data Collection

**Script**: `src/hmama/train/fetch_newsletter_data.py`

**Features**:
- Collects from 20+ reliable news sources (Indian and global)
- Uses RSS feeds for real-time news
- Extracts images from article content
- Filters recent articles (configurable time range)
- Rate limiting to avoid being blocked

**Supported Sources**:

**Indian News**:
- The Hindu
- Times of India
- Indian Express
- Hindustan Times
- Mint
- Business Standard
- NDTV
- News18
- Republic World
- Zee News

**Global News**:
- Reuters
- Associated Press
- BBC News
- CNN
- The Guardian
- New York Times
- Washington Post
- NPR
- PBS NewsHour
- Al Jazeera

**Usage**:
```bash
python src/hmama/train/fetch_newsletter_data.py \
    --output data/raw/newsletter_dataset.jsonl \
    --image_dir data/raw/newsletter_images \
    --days_back 7 \
    --max_articles_per_source 30
```

**Parameters**:
- `--days_back`: Number of days back to fetch news
- `--max_articles_per_source`: Maximum articles per source
- `--delay`: Delay between requests (rate limiting)
- `--timeout`: Request timeout in seconds

### Dataset Merging

**Script**: `src/hmama/train/merge_datasets.py`

**Features**:
- Combines Reddit, newsletter, and LIAR datasets
- Balances data across sources
- Maintains data quality and format consistency
- Configurable ratios for each source

**Usage**:
```bash
python src/hmama/train/merge_datasets.py \
    --reddit_data data/raw/reddit_dataset.jsonl \
    --newsletter_data data/raw/newsletter_dataset.jsonl \
    --liar_data data/processed/LIAR/train.jsonl \
    --output data/raw/merged_dataset.jsonl \
    --reddit_ratio 0.3 \
    --newsletter_ratio 0.4 \
    --liar_ratio 0.3
```

## 🔧 Data Format

All datasets use a consistent JSONL format:

```json
{
  "id": "unique_identifier",
  "text": "Main text content",
  "ocr": "OCR text from images",
  "image": ["path/to/image1.jpg", "path/to/image2.jpg"],
  "label": 1,
  "source": "newsletter",
  "url": "https://example.com/article",
  "created_utc": 1640995200,
  "title": "Article title",
  "summary": "Article summary"
}
```

**Fields**:
- `id`: Unique identifier
- `text`: Main text content (title + summary for newsletters)
- `ocr`: OCR text extracted from images
- `image`: List of image file paths
- `label`: 0 (FAKE) or 1 (REAL)
- `source`: Data source (reddit, newsletter, liar)
- `url`: Original URL
- `created_utc`: Unix timestamp
- `title`: Article title (newsletters only)
- `summary`: Article summary (newsletters only)

## 📊 Data Validation

**Script**: `src/hmama/train/validate_data.py`

**Features**:
- Validates data quality and format
- Checks for duplicates and missing files
- Analyzes source distribution
- Generates comprehensive reports

**Usage**:
```bash
python src/hmama/train/validate_data.py \
    --dataset data/raw/merged_dataset.jsonl \
    --check_images \
    --check_duplicates \
    --output validation_report.json
```

**Validation Checks**:
- Text quality (length, content)
- Image integrity and availability
- Duplicate content detection
- Required field validation
- Source and label distribution analysis

## 🎛️ Configuration

### Environment Variables

Set these for Reddit API access:
```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="your_user_agent"
```

### Customizing News Sources

Edit `src/hmama/train/fetch_newsletter_data.py` to add/remove news sources:

```python
NEWS_SOURCES = {
    'indian': {
        'Your Source': 'https://yoursource.com/rss',
        # ... other sources
    },
    'global': {
        'Another Source': 'https://anothersource.com/rss',
        # ... other sources
    }
}
```

## 📈 Usage Statistics

### Expected Data Volumes

- **Reddit**: 200-500 posts per run
- **Newsletter**: 300-600 articles per run (depending on sources)
- **LIAR**: ~1,000 labeled samples
- **Merged**: 1,000-2,000 balanced samples

### Performance

- **Reddit Collection**: ~5-10 minutes for 200 posts
- **Newsletter Collection**: ~15-30 minutes for 600 articles
- **Merging**: ~1-2 minutes
- **Validation**: ~2-5 minutes

## 🔍 Troubleshooting

### Common Issues

1. **Reddit API Rate Limiting**
   - Solution: Reduce `--limit` or add delays
   - Check API credentials

2. **Newsletter RSS Feed Issues**
   - Solution: Some feeds may be temporarily unavailable
   - Check internet connection and feed URLs

3. **Image Download Failures**
   - Solution: Check network connectivity
   - Some images may be blocked or removed

4. **Memory Issues with Large Datasets**
   - Solution: Reduce `--max_samples` in merge script
   - Process data in smaller batches

### Debug Mode

Run with verbose output:
```bash
python src/hmama/train/collect_all_data.py --reddit_limit 50 --newsletter_days 1
```

## 🚀 Advanced Usage

### Custom Data Collection Pipeline

```python
from src.hmama.train.fetch_reddit_dataset import main as fetch_reddit
from src.hmama.train.fetch_newsletter_data import main as fetch_newsletter
from src.hmama.train.merge_datasets import main as merge_data

# Custom collection workflow
fetch_reddit()
fetch_newsletter()
merge_data()
```

### Scheduled Data Collection

Set up cron jobs for regular data collection:
```bash
# Daily collection at 6 AM
0 6 * * * cd /path/to/h-mama-fn-xapa && python src/hmama/train/collect_all_data.py
```

## 📋 Best Practices

1. **Regular Collection**: Run data collection daily for fresh content
2. **Quality Validation**: Always validate data before training
3. **Source Diversity**: Maintain balanced representation across sources
4. **Rate Limiting**: Respect API limits and website policies
5. **Data Backup**: Keep backups of collected datasets
6. **Monitoring**: Monitor collection success rates and data quality

## 🔮 Future Enhancements

- **Real-time Collection**: Live news feed monitoring
- **Multi-language Support**: Support for regional languages
- **Advanced Filtering**: Content quality scoring
- **Automated Validation**: ML-based data quality assessment
- **Cloud Storage**: Integration with cloud storage services

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review validation reports
3. Check logs for error messages
4. Ensure all dependencies are installed

---

**The H-MaMa data collection system provides a robust, scalable solution for gathering diverse, high-quality training data for fake news detection!** 🚀
