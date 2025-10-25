#!/usr/bin/env python3
"""
Model calibration and bias analysis script
"""

import requests
import json

def test_model_bias():
    """Test the model with various types of content to identify bias patterns"""
    
    test_cases = [
        {
            "text": "Charlie Kirk gets assassinated in public.",
            "expected": "REAL",  # This should be REAL news
            "description": "Real news about public figure"
        },
        {
            "text": "Breaking: Scientists discover new planet in our solar system.",
            "expected": "REAL",
            "description": "Scientific news"
        },
        {
            "text": "SHOCKING: This one weird trick will make you rich overnight!",
            "expected": "FAKE",
            "description": "Obvious clickbait"
        },
        {
            "text": "The moon landing was faked by NASA in 1969.",
            "expected": "FAKE",
            "description": "Conspiracy theory"
        },
        {
            "text": "Local weather forecast: Sunny skies expected tomorrow.",
            "expected": "REAL",
            "description": "Neutral weather news"
        },
        {
            "text": "URGENT: Aliens have invaded Earth and are controlling our minds!",
            "expected": "FAKE",
            "description": "Obvious fake news"
        }
    ]
    
    print("🧪 Testing Model Bias and Accuracy")
    print("=" * 50)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        print(f"Expected: {test_case['expected']}")
        
        try:
            # Call the API
            response = requests.post(
                "http://localhost:8000/explain",
                json={"text": test_case["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get('label', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                explanation = result.get('human_explanation', 'No explanation')
                
                print(f"Predicted: {predicted} ({confidence}%)")
                print(f"Explanation: {explanation}")
                
                if predicted == test_case['expected']:
                    print("✅ CORRECT")
                    correct_predictions += 1
                else:
                    print("❌ INCORRECT")
                    
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n📊 Results Summary:")
    print(f"Correct: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy < 70:
        print("\n⚠️ Model accuracy is low. This could be due to:")
        print("- Training data bias")
        print("- Model overfitting to specific patterns")
        print("- Need for better feature engineering")
        print("- Insufficient training data diversity")
    
    return accuracy

def analyze_model_patterns():
    """Analyze what patterns the model is using for predictions"""
    
    print("\n🔍 Analyzing Model Patterns")
    print("=" * 40)
    
    # Test with different word patterns
    word_tests = [
        "Breaking news about Charlie Kirk",
        "Charlie Kirk news update",
        "Charlie Kirk assassination attempt",
        "Charlie Kirk public appearance",
        "Charlie Kirk controversy",
        "Charlie Kirk statement",
        "Charlie Kirk interview",
        "Charlie Kirk event"
    ]
    
    for text in word_tests:
        try:
            response = requests.post(
                "http://localhost:8000/explain",
                json={"text": text},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get('label', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                explanation = result.get('human_explanation', 'No explanation')
                
                print(f"Text: '{text}'")
                print(f"Prediction: {predicted} ({confidence}%)")
                print(f"Explanation: {explanation}")
                print("-" * 40)
                
        except Exception as e:
            print(f"Error testing '{text}': {e}")

if __name__ == "__main__":
    print("🔍 H-MaMa Model Calibration and Bias Analysis")
    print("=" * 60)
    
    # Test basic functionality
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not running. Please start it first.")
            exit(1)
    except:
        print("❌ Cannot connect to backend. Please start it first.")
        exit(1)
    
    print("✅ Backend is running")
    
    # Run bias tests
    accuracy = test_model_bias()
    
    # Analyze patterns
    analyze_model_patterns()
    
    print(f"\n🎯 Final Accuracy: {accuracy:.1f}%")
    
    if accuracy < 70:
        print("\n💡 Recommendations:")
        print("1. Retrain the model with more diverse data")
        print("2. Add more balanced training examples")
        print("3. Implement confidence thresholds")
        print("4. Add human review for low-confidence predictions")
