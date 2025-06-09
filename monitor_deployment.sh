#!/bin/bash

echo "🚀 Monitoring Render Deployment"
echo "================================"

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
sleep 60

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl -s https://waste-wise-app.onrender.com/health | jq .

# Test prediction endpoint
echo "🧪 Testing prediction endpoint..."
curl -X POST \
  -F "file=@/Users/rajendran/Desktop/Academia/Intern/waste-object-detection/model_resources/test_images/test3.jpg" \
  -F "model=YOLOv10s" \
  https://waste-wise-app.onrender.com/predict | jq .

echo "✅ Deployment test complete!" 