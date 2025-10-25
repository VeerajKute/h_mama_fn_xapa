# H-MaMa: Adaptive Multimodal Fake News Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-orange.svg)](https://pytorch.org/)

An intelligent, self-improving fake news detection system that combines text, image, and propagation analysis with adaptive learning capabilities.

## 🌟 Key Features

- **🧠 Adaptive Intelligence**: Learns from user feedback and improves over time
- **🔍 Multimodal Analysis**: Combines text, image, and social propagation data
- **📊 Dynamic Confidence**: Pattern-based confidence adjustment for better accuracy
- **🌐 Web Interface**: User-friendly Streamlit frontend with real-time feedback
- **🔄 Continuous Learning**: Self-improving system that gets better with use
- **📈 Health Monitoring**: Real-time model performance tracking and recommendations

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended)
```powershell
# Windows PowerShell
.\start_system.ps1

# Or use the Python script
python run_frontend.py
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-frontend.txt

# 3. Start backend
uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000

# 4. Start frontend (in another terminal)
streamlit run frontend.py --server.port 8501
```

### Option 3: Docker (Coming Soon)
```bash
docker-compose up
```

## 🌐 Access the System

- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 🎯 How It Works

### 1. **Multimodal Analysis**
- **Text Processing**: DistilBERT encoder with token-level attribution
- **Image Analysis**: CLIP ViT-B/16 encoder with visual saliency
- **Cross-modal Alignment**: Contradiction scoring between text and image
- **Propagation Analysis**: Social network effects and temporal patterns

### 2. **Adaptive Intelligence**
- **Pattern Recognition**: Identifies reliable sources, sensational language, clickbait
- **Dynamic Confidence**: Adjusts confidence based on content patterns and context
- **Continuous Learning**: Learns from user feedback to improve accuracy
- **Bias Detection**: Identifies and corrects model biases automatically

### 3. **Human-Readable Explanations**
- **Natural Language**: Explains predictions in plain English
- **Confidence Analysis**: Shows why confidence was adjusted
- **Uncertainty Indicators**: Highlights what makes predictions uncertain
- **Counterfactual Reasoning**: Explains what would change the prediction

## 📊 Data Preparation

### Option A: Use Demo Data
```bash
# Fetch Reddit data
python src/hmama/train/fetch_reddit_dataset.py --limit 50 \
  --output src/hmama/train/reddit_dataset.jsonl --image_dir src/hmama/train/reddit_images

# Convert to single-image format
python src/hmama/train/convert_reddit_to_single.py --input src/hmama/train/reddit_dataset.jsonl \
  --output data/reddit_single.jsonl
```

### Option B: Custom Data
- **Single-image format**:
```json
{"id": "123", "text": "News content...", "image_path": "path/to.jpg", "label": 0}
```
- **Multi-image format**:
```json
{"id": "123", "text": "News content...", "image": ["path/a.jpg", "path/b.jpg"], "label": 0}
```

### Option C: Use Existing Datasets
```bash
# LIAR dataset
python src/hmama/scripts/liar_to_jsonl.py --input data/raw/LIAR --output data/processed/LIAR

# FakeNewsNet dataset
python src/hmama/scripts/fakenewsnet_to_jsonl.py --input data/raw/FakeNewsNet --output data/raw/fakenewsnet.jsonl
```

## 🏋️ Training

### Basic Training
```bash
python src/hmama/train/train_full.py --data data/reddit_single.jsonl --epochs 1 --batch 4 --device cuda
```

### With Evaluation
```bash
python src/hmama/train/train_full.py --data data/reddit_single.jsonl --eval_data data/reddit_single.jsonl --epochs 3
```

### Advanced Training with Propagation
```bash
python src/hmama/train/train_full.py --data data/raw/fakenewsnet.jsonl --propagation data/raw/fakenewsnet_propagation.jsonl --epochs 3 --batch 8
```

## 🔧 API Endpoints

### Core Endpoints
- **`POST /predict`**: Basic prediction with confidence scores
- **`POST /explain`**: Detailed explanation with human-readable justifications
- **`POST /feedback`**: Submit corrections to improve the model
- **`GET /model_health`**: Model performance and health metrics
- **`GET /health`**: System health check

### Example API Usage
```python
import requests

# Basic prediction
response = requests.post("http://localhost:8000/predict", json={
    "text": "Breaking news about climate change",
    "image_path": "path/to/image.jpg"  # Optional
})

# Get explanation
response = requests.post("http://localhost:8000/explain", json={
    "text": "Breaking news about climate change",
    "image_path": "path/to/image.jpg"  # Optional
})

# Submit feedback
response = requests.post("http://localhost:8000/feedback", json={
    "text": "Breaking news about climate change",
    "predicted_label": "FAKE",
    "predicted_confidence": 0.85,
    "actual_label": "REAL"
})
```

## 🧪 Testing & Evaluation

### Test the Adaptive System
```bash
# Comprehensive system test
python test_adaptive_system.py

# Model calibration test
python model_calibration.py

# API functionality test
python test_api.py
```

### Model Health Monitoring
```bash
# Check model health
curl http://localhost:8000/model_health

# Get detailed metrics
python -c "
import requests
response = requests.get('http://localhost:8000/model_health')
print(response.json())
"
```

## 🔄 Continuous Learning

The system learns from user feedback to improve accuracy:

1. **User Feedback**: Users can correct wrong predictions through the web interface
2. **Pattern Learning**: System identifies patterns in corrections
3. **Confidence Adjustment**: Dynamically adjusts confidence thresholds
4. **Model Updates**: Continuously improves prediction accuracy

### Feedback Loop
```
User Input → Model Prediction → User Feedback → System Learning → Improved Predictions
```

## 📈 Performance Monitoring

### Model Health Metrics
- **Health Score**: Overall model performance (0-1)
- **Reliability**: Consistency of predictions (0-1)
- **Uncertainty**: Model confidence in its predictions (0-1)
- **Feedback Count**: Number of user corrections received

### Confidence Analysis
- **Pattern Recognition**: Identifies reliable sources, sensational language
- **Context Awareness**: Considers text length, punctuation, emotional content
- **Dynamic Adjustment**: Adapts confidence based on content patterns
- **Uncertainty Indicators**: Shows what makes predictions uncertain

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Reddit API (for data collection)
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="H-MaMa Data Collector"

# Model Configuration
export MODEL_DEVICE="cuda"  # or "cpu"
export CONFIDENCE_THRESHOLD="0.7"
export LEARNING_RATE="0.001"
```

### Custom Model Training
```bash
# Train with custom parameters
python src/hmama/train/train_full.py \
  --data data/custom_dataset.jsonl \
  --epochs 10 \
  --batch 16 \
  --lr 0.0001 \
  --device cuda \
  --save_dir checkpoints/custom_model
```

## 📁 Project Structure

```
h-mama-fn-xapa/
├── src/hmama/                    # Core source code
│   ├── models/                   # Model implementations
│   │   ├── adaptive_calibration.py  # Learning system
│   │   ├── dynamic_confidence.py    # Confidence adjustment
│   │   ├── text_encoder_full.py     # Text processing
│   │   ├── clip_encoder_ft.py       # Image processing
│   │   └── ug_mop_adaptive.py       # Fusion model
│   ├── serve/                    # API server
│   │   └── api.py               # FastAPI endpoints
│   ├── train/                    # Training scripts
│   ├── data/                     # Data processing
│   └── explain/                  # Explanation utilities
├── frontend.py                   # Streamlit web interface
├── test_adaptive_system.py       # System tests
├── model_calibration.py          # Bias testing
├── start_system.ps1             # Windows startup script
├── requirements.txt              # Backend dependencies
├── requirements-frontend.txt     # Frontend dependencies
└── checkpoints/                  # Model checkpoints
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Submit a pull request** with detailed description

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-frontend.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black src/ frontend.py

# Lint code
flake8 src/ frontend.py
```

## 🙏 Acknowledgments

- **DistilBERT**: For efficient text encoding
- **CLIP**: For multimodal understanding
- **FastAPI**: For robust API framework
- **Streamlit**: For interactive web interface
- **PyTorch**: For deep learning framework

## 🔮 Roadmap

- [ ] **Multi-language Support**: Detect fake news in multiple languages
- [ ] **Real-time Monitoring**: Live news feed analysis
- [ ] **Mobile App**: Native mobile application
- [ ] **Advanced Visualizations**: Interactive dashboards
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Docker Deployment**: Containerized production setup
- [ ] **Cloud Integration**: AWS/Azure deployment options
- [ ] **API Rate Limiting**: Production-ready API management

---

**⭐ Star this repository if you find it helpful!**
