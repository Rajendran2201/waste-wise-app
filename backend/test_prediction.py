#!/usr/bin/env python3
"""
Test script to verify prediction endpoint works with custom modules
"""

import requests
import os
import time

def test_prediction():
    """Test the prediction endpoint"""
    print("🧪 Testing prediction endpoint...")
    
    # Test image path
    test_image = "/Users/rajendran/Desktop/Academia/Intern/waste-object-detection/model_resources/test_images/test3.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    # Test local endpoint
    local_url = "http://127.0.0.1:5001/predict"
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'model': 'YOLOv10s'}
            
            print(f"📤 Sending request to {local_url}")
            response = requests.post(local_url, files=files, data=data)
            
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction successful!")
                print(f"📊 Detections: {len(result.get('detections', []))}")
                print(f"🖼️ Result image: {result.get('result_image', 'N/A')}")
                return True
            else:
                print(f"❌ Prediction failed: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_deployed_prediction():
    """Test the deployed prediction endpoint"""
    print("\n🧪 Testing deployed prediction endpoint...")
    
    # Test image path
    test_image = "/Users/rajendran/Desktop/Academia/Intern/waste-object-detection/model_resources/test_images/test3.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    # Test deployed endpoint
    deployed_url = "https://waste-wise-app.onrender.com/predict"
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'model': 'YOLOv10s'}
            
            print(f"📤 Sending request to {deployed_url}")
            response = requests.post(deployed_url, files=files, data=data)
            
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Deployed prediction successful!")
                print(f"📊 Detections: {len(result.get('detections', []))}")
                print(f"🖼️ Result image: {result.get('result_image', 'N/A')}")
                return True
            else:
                print(f"❌ Deployed prediction failed: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Deployed request failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Prediction Endpoint Test")
    print("=" * 50)
    
    # Test local endpoint
    local_ok = test_prediction()
    
    # Test deployed endpoint
    deployed_ok = test_deployed_prediction()
    
    print("\n" + "=" * 50)
    if local_ok and deployed_ok:
        print("✅ All tests passed! Your prediction endpoint is working.")
    elif local_ok:
        print("⚠️ Local works but deployed needs more time or has issues.")
    else:
        print("❌ Tests failed. Check the errors above.") 