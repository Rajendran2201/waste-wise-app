# WasteWise - AI Waste Object Detection

An AI-powered waste object detection application using YOLO models (v10, v11, v12) for real-time waste classification and recycling guidance.

## 🚀 Features

- **Multi-Model Support**: YOLOv10, YOLOv11, YOLOv12
- **Multiple Input Types**: Images, Videos, Live Webcam
- **Real-time Detection**: Live object detection with bounding boxes
- **Video Processing**: Process videos with object detection overlay
- **Webcam Support**: Real-time detection using your camera
- **Modern UI**: Beautiful, responsive React interface

## 🛠️ Tech Stack

### Backend
- **Flask**: Python web framework
- **OpenCV**: Computer vision processing
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO model integration
- **Gunicorn**: Production WSGI server

### Frontend
- **React**: Modern JavaScript framework
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful icons
- **HTML5 Video**: Video playback support

## 📦 Installation

### Prerequisites
- Python 3.11+
- Node.js 16+
- FFmpeg (for video processing)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend/waste-detection-ui
npm install
npm start
```

## 🚀 Deployment

### Option 1: Render.com (Recommended for Free)

#### Backend Deployment
1. **Create Render Account**: Sign up at [render.com](https://render.com)
2. **New Web Service**: 
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
3. **Environment Variables**:
   - `PORT`: 10000 (auto-set by Render)
4. **Upload Models**: Add your YOLO model files to `backend/models/`

#### Frontend Deployment
1. **New Static Site**:
   - Build Command: `npm run build`
   - Publish Directory: `build`
2. **Environment Variables**:
   - `REACT_APP_API_URL`: Your backend URL

### Option 2: Railway.app

#### Backend
1. **Connect Repository**: Link your GitHub repo
2. **Auto-deploy**: Railway detects Python and deploys
3. **Environment**: Add your model files

#### Frontend
1. **Static Site**: Deploy React build
2. **Environment**: Set API URL

### Option 3: Vercel (Frontend Only)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd frontend/waste-detection-ui
vercel
```

## 🔧 Configuration

### Environment Variables

#### Backend
- `PORT`: Server port (default: 5001)
- `DEBUG`: Debug mode (default: False)

#### Frontend
- `REACT_APP_API_URL`: Backend API URL

### Model Files
Place your YOLO model files in `backend/models/`:
- `YOLOv10s.pt`
- `YOLOv10m.pt`
- `YOLOv11n.pt`
- `YOLOv12s.pt`

## 📁 Project Structure

```
waste-object-detection-app/
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── Procfile              # Deployment configuration
│   ├── runtime.txt           # Python version
│   ├── models/               # YOLO model files
│   ├── utils/                # Utility functions
│   ├── custom_yolo_modules.py # Custom YOLO modules
│   └── static/results/       # Processed results
├── frontend/
│   └── waste-detection-ui/
│       ├── src/
│       │   ├── App.js        # Main React component
│       │   └── config.js     # Environment configuration
│       ├── public/           # Static files
│       └── package.json      # Node.js dependencies
└── README.md
```

## 🎯 Usage

1. **Image Detection**: Upload images for waste object detection
2. **Video Processing**: Upload videos for frame-by-frame analysis
3. **Live Webcam**: Use your camera for real-time detection
4. **Model Selection**: Choose between YOLOv10, YOLOv11, or YOLOv12

## 🔒 Security Considerations

- **File Upload Limits**: Implement file size restrictions
- **CORS Configuration**: Configure allowed origins
- **Rate Limiting**: Add request rate limiting
- **Input Validation**: Validate all user inputs

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model files are in correct location
2. **Video Playback Issues**: Check FFmpeg installation
3. **CORS Errors**: Verify backend CORS configuration
4. **Memory Issues**: Reduce video resolution or use smaller models

### Debug Mode
```bash
# Backend
export FLASK_ENV=development
python app.py

# Frontend
npm start
```

## 📈 Performance Optimization

- **Model Optimization**: Use quantized models
- **Video Compression**: Implement video compression
- **Caching**: Add result caching
- **CDN**: Use CDN for static assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Ultralytics for YOLO models
- OpenCV for computer vision
- React and Tailwind CSS communities
