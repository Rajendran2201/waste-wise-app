#!/bin/bash

echo "🚀 Deploying WasteWise Frontend..."

# Navigate to frontend directory
cd frontend/waste-detection-ui

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build the project
echo "🔨 Building React app..."
npm run build

# Check if build was successful
if [ ! -d "build" ]; then
    echo "❌ Build failed! Please check for errors."
    exit 1
fi

echo "✅ Build successful!"

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
echo ""
echo "📋 Vercel deployment steps:"
echo "1. Run: npx vercel"
echo "2. Follow the prompts:"
echo "   - Set up and deploy? → Y"
echo "   - Project name → wastewise-frontend"
echo "   - Directory → ./ (current)"
echo "3. Add environment variable:"
echo "   - Name: REACT_APP_API_URL"
echo "   - Value: https://waste-wise-app.onrender.com"
echo ""
echo "🔗 Your backend is ready at: https://waste-wise-app.onrender.com"
echo "🔗 Your frontend will be at: https://wastewise-frontend.vercel.app"
echo ""
echo "💡 Alternative: Use 'npx vercel --prod' for production deployment"

# Offer to run vercel command
read -p "Would you like to run 'npx vercel' now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    npx vercel
fi 