#!/bin/bash

echo "🔍 Checking deployment status..."
echo "=================================="

# Check if the backend is responding
echo "Testing backend health endpoint..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" https://waste-wise-app.onrender.com/health

echo ""
echo "Testing backend with timeout (30s)..."
timeout 30s curl -s https://waste-wise-app.onrender.com/health

echo ""
echo "=================================="
echo "If you see HTTP 200, the deployment is successful!"
echo "If you see connection errors, the deployment is still in progress."
echo ""
echo "You can also check the deployment logs at:"
echo "https://dashboard.render.com/web/sites/waste-wise-app" 