#!/usr/bin/env python3
"""
Test script to verify the API works with text-only requests
"""

import requests
import json

def test_text_only():
    """Test API with text-only request"""
    print("🧪 Testing text-only API request...")
    
    payload = {
        "text": "Charlie Kirk gets assassinated in public."
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/explain",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Label: {result.get('label')}")
            print(f"Confidence: {result.get('confidence')}%")
            print(f"Explanation: {result.get('human_explanation')}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure backend is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_image():
    """Test API with image request"""
    print("\n🧪 Testing API with image request...")
    
    payload = {
        "text": "Charlie Kirk gets assassinated in public.",
        "image_path": "data/raw/images/img1.jpg"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/explain",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call with image successful!")
            print(f"Label: {result.get('label')}")
            print(f"Confidence: {result.get('confidence')}%")
            print(f"Explanation: {result.get('human_explanation')}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure backend is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 H-MaMa API Test")
    print("=" * 30)
    
    # Test text-only
    text_success = test_text_only()
    
    # Test with image
    image_success = test_with_image()
    
    print("\n📊 Test Results:")
    print(f"Text-only: {'✅' if text_success else '❌'}")
    print(f"With image: {'✅' if image_success else '❌'}")
    
    if text_success and image_success:
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
