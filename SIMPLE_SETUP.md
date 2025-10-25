# 🚀 H-MaMa - One Command Setup

**The simplest way to get H-MaMa fake news detection running!**

## ⚡ Quick Start (One Command)

### Windows Users
```bash
# Double-click this file or run in command prompt:
run_h_mama.bat

# Or in PowerShell:
.\run_h_mama.ps1
```

### Mac/Linux Users
```bash
python run_h_mama.py
```

## 🎯 What This Does

The `run_h_mama.py` script automatically:

1. **📦 Installs Everything**
   - Creates virtual environment
   - Installs all required packages
   - Sets up dependencies

2. **📊 Collects Data**
   - Scrapes Reddit posts (news, worldnews, technology)
   - Fetches recent news from reliable sources
   - Merges all data into training dataset

3. **🧠 Trains Model**
   - Trains fake news detection model
   - Uses collected data for learning
   - Saves trained model

4. **🌐 Starts Web Interface**
   - Launches web interface at http://localhost:8501
   - Ready to detect fake news!

## ⏱️ Time Required

- **First run**: 10-15 minutes (includes installation)
- **Subsequent runs**: 5-10 minutes (just data collection + training)

## 🔧 Requirements

- Python 3.8+ installed
- Internet connection
- 2GB free disk space

## 🎮 Usage

1. **Run the script**: `python run_h_mama.py`
2. **Wait for setup**: Let it install and collect data
3. **Use the web interface**: Open http://localhost:8501
4. **Test fake news detection**: Paste text or upload images

## 🛠️ Troubleshooting

### If something fails:
- Check internet connection
- Make sure Python 3.8+ is installed
- Try running as administrator (Windows)
- Check if antivirus is blocking the script

### If data collection fails:
- The script will create a fallback dataset
- You can still train and test the model
- Try again later for fresh data

## 📱 What You Get

- **Web Interface**: Easy-to-use fake news detector
- **Text Analysis**: Paste any text to check if it's fake
- **Image Analysis**: Upload images with text
- **Confidence Scores**: See how confident the model is
- **Real-time Learning**: Model improves with use

## 🎉 That's It!

No complex setup, no multiple commands, no configuration files. Just run one script and you're ready to detect fake news!

---

**Ready to start? Just run: `python run_h_mama.py`** 🚀
