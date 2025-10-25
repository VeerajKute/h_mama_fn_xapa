# H-MaMa Fake News Detection Frontend

A user-friendly Streamlit web interface for the H-MaMa multimodal fake news detection system.

## Features

- **Text Analysis**: Enter news text for fake news detection
- **Image Upload**: Optional image upload for multimodal analysis
- **Real-time Results**: Get instant predictions with confidence scores
- **Human-readable Explanations**: Understand why the model made its decision
- **Visual Feedback**: Color-coded results and clean interface
- **Error Handling**: Graceful handling of API errors and missing files

## Quick Start

### Option 1: Automated Startup (Recommended)
```bash
# Install frontend dependencies
pip install -r requirements-frontend.txt

# Run the automated startup script
python run_frontend.py
```

This will:
1. Check if the backend is running
2. Start the backend if needed
3. Launch the Streamlit frontend

### Option 2: Manual Startup

1. **Start the Backend** (in one terminal):
```bash
uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000
```

2. **Start the Frontend** (in another terminal):
```bash
pip install -r requirements-frontend.txt
streamlit run frontend.py --server.port 8501
```

3. **Open your browser** to: http://localhost:8501

## Usage

1. **Enter Text**: Type or paste the news article you want to analyze
2. **Upload Image** (Optional): Upload an image related to the news
3. **Click "Check for Fake News"**: Analyze the content
4. **View Results**: See the prediction, confidence, and explanation

## Interface Components

### Main Input Area
- **Text Input**: Large text area for news content
- **Image Upload**: File uploader for images (PNG, JPG, JPEG, GIF, BMP)
- **Image Preview**: Shows uploaded image before analysis

### Results Display
- **Prediction Box**: Color-coded FAKE (red) or REAL (green) with confidence
- **Explanation**: Human-readable explanation of the model's decision
- **Technical Details**: Expandable section with detailed metrics

### Sidebar
- **Instructions**: Step-by-step usage guide
- **About**: Information about the system and features

## API Integration

The frontend communicates with the FastAPI backend at `http://localhost:8000`:

- **Endpoint**: `POST /explain`
- **Payload**: `{"text": "news text", "image_path": "path/to/image.jpg"}`
- **Response**: Prediction, confidence, explanation, and technical details

## Error Handling

The frontend handles various error scenarios:

- **API Unreachable**: Shows connection error with instructions
- **Missing Image**: Gracefully handles missing image files
- **Invalid Image**: Shows error for corrupted image files
- **Empty Text**: Prevents submission without text input
- **Timeout**: Handles slow API responses

## File Structure

```
├── frontend.py                 # Main Streamlit application
├── run_frontend.py            # Automated startup script
├── requirements-frontend.txt  # Frontend dependencies
├── FRONTEND_README.md         # This file
└── temp_images/              # Temporary image storage (auto-created)
```

## Dependencies

- **streamlit**: Web interface framework
- **requests**: HTTP client for API calls
- **Pillow**: Image processing and display

## Customization

### Styling
The interface uses custom CSS for styling. You can modify the styles in the `st.markdown()` section at the top of `frontend.py`.

### API Endpoint
To change the backend URL, modify the `call_explain_api()` function in `frontend.py`:
```python
response = requests.post(
    "http://your-backend-url:port/explain",  # Change this
    json=payload,
    timeout=30
)
```

### Image Handling
Images are temporarily saved in the `temp_images/` directory. You can modify the `save_uploaded_file()` function to change the storage location or add cleanup logic.

## Troubleshooting

### Common Issues

1. **"Cannot connect to API"**
   - Ensure the backend is running on port 8000
   - Check if the backend URL is correct
   - Verify firewall settings

2. **"Image upload failed"**
   - Check file permissions for the temp_images directory
   - Ensure the image file is not corrupted
   - Try a different image format

3. **"Streamlit not found"**
   - Install dependencies: `pip install -r requirements-frontend.txt`
   - Check Python environment

4. **"Port already in use"**
   - Change the port in the startup command
   - Kill existing processes using the port

### Debug Mode

To run Streamlit in debug mode:
```bash
streamlit run frontend.py --server.port 8501 --logger.level debug
```

## Development

### Adding New Features

1. **New Input Fields**: Add to the main form in `main()`
2. **New Results Display**: Modify `display_prediction_result()`
3. **New API Calls**: Add functions similar to `call_explain_api()`

### Testing

Test the frontend with different scenarios:
- Text-only analysis
- Text + image analysis
- Invalid images
- Empty text
- API errors

## Support

For issues with the frontend, check:
1. Backend is running and accessible
2. All dependencies are installed
3. File permissions are correct
4. Network connectivity

For backend issues, refer to the main project documentation.
