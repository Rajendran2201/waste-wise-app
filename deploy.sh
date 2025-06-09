#!/bin/bash

# WasteWise Deployment Script
echo "🚀 Starting WasteWise Deployment..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Build frontend
echo "🔨 Building frontend..."
cd frontend/waste-detection-ui
npm run build
cd ../..

# Check if backend models exist
echo "🔍 Checking backend models..."
if [ ! -d "backend/models" ]; then
    echo "⚠️  Warning: backend/models directory not found!"
    echo "Please add your YOLO model files to backend/models/"
    echo "Required files: YOLOv10s.pt, YOLOv10m.pt, YOLOv11n.pt, YOLOv12s.pt"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# React
frontend/waste-detection-ui/build/
frontend/waste-detection-ui/.env.local
frontend/waste-detection-ui/.env.development.local
frontend/waste-detection-ui/.env.test.local
frontend/waste-detection-ui/.env.production.local

# Backend
backend/uploads/
backend/static/results/
*.log

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
EOF
fi

echo "✅ Deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Push to GitHub: git push origin main"
echo "2. Deploy backend to Render/Railway"
echo "3. Deploy frontend to Vercel/Netlify"
echo "4. Update REACT_APP_API_URL in frontend environment"
echo ""
echo "🔗 Deployment URLs:"
echo "- Backend: https://your-app-name.onrender.com"
echo "- Frontend: https://your-app-name.vercel.app" 