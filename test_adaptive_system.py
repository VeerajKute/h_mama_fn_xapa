#!/usr/bin/env python3
"""
Test script for the adaptive fake news detection system
"""

import requests
import json
import time
from typing import Dict, List

def test_adaptive_system():
    """Test the adaptive system with various scenarios"""
    
    print("🧪 Testing Adaptive Fake News Detection System")
    print("=" * 60)
    
    # Test cases with expected outcomes
    test_cases = [
        {
            "text": "Charlie Kirk gets assassinated in public.",
            "expected": "REAL",
            "description": "Real news about public figure",
            "should_learn": True
        },
        {
            "text": "SHOCKING: This one weird trick will make you rich overnight!",
            "expected": "FAKE",
            "description": "Obvious clickbait",
            "should_learn": True
        },
        {
            "text": "According to Reuters, scientists have discovered a new planet.",
            "expected": "REAL",
            "description": "News with reliable source",
            "should_learn": True
        },
        {
            "text": "BREAKING: Aliens have invaded Earth and are controlling our minds!",
            "expected": "FAKE",
            "description": "Obvious fake news",
            "should_learn": True
        }
    ]
    
    print("📊 Running initial predictions...")
    initial_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        
        try:
            # Get prediction
            response = requests.post(
                "http://localhost:8000/explain",
                json={"text": test_case["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get('label', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                raw_confidence = result.get('raw_confidence', 0)
                adjustment_factors = result.get('adjustment_factors', {})
                uncertainty_indicators = result.get('uncertainty_indicators', [])
                
                print(f"Predicted: {predicted} ({confidence}%)")
                print(f"Raw confidence: {raw_confidence}%")
                print(f"Expected: {test_case['expected']}")
                
                # Show adjustment factors
                if adjustment_factors:
                    print("Adjustment factors:")
                    for factor, value in adjustment_factors.items():
                        print(f"  {factor}: {value:.3f}")
                
                # Show uncertainty indicators
                if uncertainty_indicators:
                    print("Uncertainty indicators:")
                    for indicator in uncertainty_indicators:
                        print(f"  • {indicator}")
                
                # Store result
                initial_results.append({
                    'test_case': test_case,
                    'predicted': predicted,
                    'confidence': confidence,
                    'raw_confidence': raw_confidence,
                    'adjustment_factors': adjustment_factors
                })
                
                # Submit feedback if prediction is wrong
                if predicted != test_case['expected'] and test_case['should_learn']:
                    print(f"❌ Wrong prediction - submitting feedback...")
                    submit_feedback(test_case['text'], predicted, confidence/100, test_case['expected'])
                    time.sleep(1)  # Wait for feedback processing
                
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test model health
    print("\n📊 Checking model health...")
    try:
        health_response = requests.get("http://localhost:8000/model_health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Model health: {health_data.get('health_metrics', {})}")
            print(f"Recommendations: {health_data.get('recommendations', [])}")
            print(f"Feedback count: {health_data.get('feedback_count', 0)}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test learning by running predictions again
    print("\n🔄 Testing learning by running predictions again...")
    for i, result in enumerate(initial_results, 1):
        test_case = result['test_case']
        print(f"\n📝 Re-test {i}: {test_case['description']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/explain",
                json={"text": test_case["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                new_result = response.json()
                new_predicted = new_result.get('label', 'UNKNOWN')
                new_confidence = new_result.get('confidence', 0)
                
                print(f"Original: {result['predicted']} ({result['confidence']}%)")
                print(f"New: {new_predicted} ({new_confidence}%)")
                
                if new_predicted != result['predicted']:
                    print("✅ Model prediction changed - learning detected!")
                elif abs(new_confidence - result['confidence']) > 5:
                    print("✅ Model confidence adjusted - learning detected!")
                else:
                    print("ℹ️ No significant change detected")
                    
        except Exception as e:
            print(f"❌ Error in re-test: {e}")

def submit_feedback(text, predicted_label, predicted_confidence, actual_label):
    """Submit feedback to the model"""
    try:
        payload = {
            "text": text,
            "predicted_label": predicted_label,
            "predicted_confidence": predicted_confidence,
            "actual_label": actual_label,
            "user_id": "test_user"
        }
        
        response = requests.post(
            "http://localhost:8000/feedback",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Feedback submitted: {result.get('message', '')}")
            return True
        else:
            print(f"❌ Feedback failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        return False

def test_confidence_adjustments():
    """Test confidence adjustment patterns"""
    print("\n🔍 Testing Confidence Adjustment Patterns")
    print("=" * 50)
    
    # Test different text patterns
    pattern_tests = [
        {
            "text": "According to Reuters, the economy is growing.",
            "description": "Reliable source + factual language"
        },
        {
            "text": "SHOCKING BREAKING NEWS!!! You won't believe this!",
            "description": "Sensational language + excessive punctuation"
        },
        {
            "text": "The weather today is sunny with a chance of rain.",
            "description": "Neutral, factual content"
        },
        {
            "text": "URGENT: Click here to see what doctors don't want you to know!",
            "description": "Clickbait + urgency language"
        }
    ]
    
    for test in pattern_tests:
        print(f"\n📝 Testing: {test['description']}")
        print(f"Text: '{test['text']}'")
        
        try:
            response = requests.post(
                "http://localhost:8000/explain",
                json={"text": test["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get('label', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                raw_confidence = result.get('raw_confidence', 0)
                adjustment_factors = result.get('adjustment_factors', {})
                confidence_explanation = result.get('confidence_explanation', '')
                
                print(f"Predicted: {predicted} ({confidence}%)")
                print(f"Raw confidence: {raw_confidence}%")
                print(f"Confidence explanation: {confidence_explanation}")
                
                if adjustment_factors:
                    print("Adjustment factors:")
                    for factor, value in adjustment_factors.items():
                        print(f"  {factor}: {value:.3f}")
                        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Adaptive System Tests")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not running. Please start it first.")
            exit(1)
    except:
        print("❌ Cannot connect to backend. Please start it first.")
        exit(1)
    
    print("✅ Backend is running")
    
    # Run tests
    test_adaptive_system()
    test_confidence_adjustments()
    
    print("\n🎉 Adaptive system testing complete!")
    print("The model should now be more accurate and adaptive to different content patterns.")
