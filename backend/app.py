from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import threading
import time
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for webcam
camera = None
camera_lock = threading.Lock()
current_model = None
current_model_name = None

# CRITICAL: Set up custom modules BEFORE any other imports
try:
    from startup import setup_custom_modules, patch_ultralytics_directly
    
    # Try setup first
    if not setup_custom_modules():
        # Fallback to direct patching
        patch_ultralytics_directly()
    
    logger.info("✅ Custom YOLO modules loaded and registered successfully")
except Exception as e:
    logger.error(f"❌ Failed to set up custom modules: {e}")

# Now import inference utilities
from utils.inference_with_custom import load_model, run_inference
from utils.video_inference import run_video_inference

app = Flask(__name__)

# Configure CORS to allow all origins
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def initialize_camera():
    """Initialize the webcam"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            # Try different camera indices
            for i in range(1, 4):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    break
        if not camera.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        print("✅ Webcam initialized successfully")

def release_camera():
    """Release the webcam"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("✅ Webcam released")

def load_model_for_webcam(model_name):
    """Load model for webcam inference"""
    global current_model, current_model_name
    if current_model_name != model_name:
        try:
            current_model, method = load_model(model_name)
            current_model_name = model_name
            print(f"✅ Model {model_name} loaded for webcam using {method}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return False
    return True

def run_inference_on_frame(frame, model):
    """Run inference on a single frame"""
    try:
        if model is None:
            return frame
        
        # Run inference using ultralytics
        results = model(frame)
        result = results[0]
        
        # Get the annotated frame
        annotated_frame = result.plot()
        
        return annotated_frame
    except Exception as e:
        print(f"❌ Frame inference error: {e}")
        # Return original frame if inference fails
        return frame

def generate_frames(model_name="YOLOv10s"):
    """Generate frames for webcam stream with real-time detection"""
    global camera, current_model
    
    try:
        # Initialize camera if not already done
        if camera is None:
            initialize_camera()
        
        # Load model
        if not load_model_for_webcam(model_name):
            # If model loading fails, show error message on frame
            while True:
                success, frame = camera.read()
                if not success:
                    break
                
                # Create error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Model {model_name} failed to load", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Main inference loop
        while True:
            with camera_lock:
                success, frame = camera.read()
            
            if not success:
                print("❌ Failed to read frame from webcam")
                break
            
            # Run inference on frame
            annotated_frame = run_inference_on_frame(frame, current_model)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("❌ Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        print(f"❌ Webcam stream error: {e}")
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Webcam Error: {str(e)}", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/predict_webcam')
def predict_webcam():
    """Real-time webcam detection endpoint"""
    model_name = request.args.get('model', 'YOLOv10s')
    
    # Validate model name
    valid_models = ["YOLOv10s", "YOLOv10m", "YOLOv11n", "YOLOv12s"]
    if model_name not in valid_models:
        return jsonify({"error": f"Invalid model. Choose from: {valid_models}"}), 400
    
    try:
        return Response(
            stream_with_context(generate_frames(model_name)),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        return jsonify({"error": f"Webcam stream failed: {str(e)}"}), 500

@app.route('/webcam/start')
def start_webcam():
    """Start webcam and return status"""
    try:
        initialize_camera()
        return jsonify({"status": "success", "message": "Webcam started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webcam/stop')
def stop_webcam():
    """Stop webcam and return status"""
    try:
        release_camera()
        return jsonify({"status": "success", "message": "Webcam stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webcam/status')
def webcam_status():
    """Check webcam status"""
    global camera
    status = {
        "camera_initialized": camera is not None and camera.isOpened(),
        "current_model": current_model_name,
        "model_loaded": current_model is not None
    }
    return jsonify(status)

@app.route("/predict_video", methods=["POST"])
def predict_video():
    """Handle video file uploads and processing"""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        model_name = request.form.get('model')

        if file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        if not model_name:
            return jsonify({"error": "No model specified"}), 400

        # Validate file type
        ext = os.path.splitext(file.filename)[-1].lower()
        valid_video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        if ext not in valid_video_extensions:
            return jsonify({"error": f"Invalid video format. Supported formats: {valid_video_extensions}"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process video
        result_path = run_video_inference(filepath, model_name)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Determine the file extension for proper MIME type
        file_ext = os.path.splitext(result_path)[-1].lower()
        content_type = "video/mp4" if file_ext == ".mp4" else "video/avi"
        
        return jsonify({
            "video_url": f"/results/{os.path.basename(result_path)}",
            "results": "Video processed successfully",
            "message": f"Video processed with {model_name} model",
            "file_type": content_type,
            "file_extension": file_ext
        })

    except Exception as e:
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image file uploads and processing with enhanced error handling and memory management"""
    try:
        logger.info("📥 Received prediction request")
        
        # Check if file is present
        if 'file' not in request.files:
            logger.error("❌ No file provided in request")
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        model_name = request.form.get('model', 'YOLOv10s')
        
        logger.info(f"📋 Request details - Model: {model_name}, Filename: {file.filename}")

        if file.filename == '':
            logger.error("❌ No file selected")
            return jsonify({"error": "No file selected"}), 400
        if not model_name:
            logger.error("❌ No model specified")
            return jsonify({"error": "No model specified"}), 400

        # Memory optimization: Clear cache before processing
        try:
            from utils.inference_with_custom import clear_model_cache
            clear_model_cache()
        except:
            pass

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"💾 Saving file to: {filepath}")
        
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            logger.error(f"❌ File was not saved properly: {filepath}")
            return jsonify({"error": "Failed to save uploaded file"}), 500
            
        file_size = os.path.getsize(filepath)
        logger.info(f"✅ File saved successfully - Size: {file_size} bytes")

        ext = os.path.splitext(filename)[-1].lower()
        logger.info(f"📄 File extension: {ext}")

        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            logger.info("🎬 Processing video file")
            result_path = run_video_inference(filepath, model_name)
            os.remove(filepath)
            logger.info(f"✅ Video processing complete: {result_path}")
            return jsonify({
                "video_url": f"/results/{os.path.basename(result_path)}",
                "results": "Video processed successfully"
            })
        else:
            logger.info("🖼️ Processing image file")
            
            # Add timeout for model loading
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Model loading timed out")
            
            # Set timeout for model loading (60 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                result_path, result_data = run_inference(filepath, model_name)
                signal.alarm(0)  # Cancel timeout
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                logger.error("❌ Model loading timed out")
                return jsonify({"error": "Model loading timed out. Please try a smaller model."}), 500
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                raise e
            
            os.remove(filepath)
            logger.info(f"✅ Image processing complete - Detections: {len(result_data)}")
            return jsonify({
                "image_url": f"/results/{os.path.basename(result_path)}",
                "results": result_data
            })

    except Exception as e:
        logger.error(f"❌ Prediction failed with error: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Clean up any uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route("/results/<filename>")
def get_result(filename):
    """Serve result files (images and videos) with proper headers"""
    file_path = os.path.join(RESULT_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Set proper headers based on file type
    ext = os.path.splitext(filename)[-1].lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Video files - set proper headers for streaming
        response = send_from_directory(RESULT_FOLDER, filename)
        
        # Set appropriate content type based on file extension
        if ext == '.mp4':
            response.headers['Content-Type'] = 'video/mp4'
        elif ext == '.avi':
            response.headers['Content-Type'] = 'video/avi'
        elif ext == '.mov':
            response.headers['Content-Type'] = 'video/quicktime'
        elif ext == '.mkv':
            response.headers['Content-Type'] = 'video/x-matroska'
        elif ext == '.webm':
            response.headers['Content-Type'] = 'video/webm'
        
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        return response
    else:
        # Image files
        return send_from_directory(RESULT_FOLDER, filename)

@app.route("/debug/video/<filename>")
def debug_video(filename):
    """Debug endpoint to check video file accessibility"""
    file_path = os.path.join(RESULT_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found", "path": file_path}), 404
    
    file_stats = os.stat(file_path)
    
    return jsonify({
        "exists": True,
        "path": file_path,
        "size": file_stats.st_size,
        "size_mb": file_stats.st_size / 1024 / 1024,
        "readable": os.access(file_path, os.R_OK),
        "content_type": "video/mp4" if filename.endswith('.mp4') else "video/avi"
    })

@app.route("/health")
def health_check():
    """Health check endpoint to verify setup"""
    try:
        # Check if custom modules are loaded
        import custom_yolo_modules
        custom_loaded = True
    except:
        custom_loaded = False
    
    # Check model files
    model_files = {}
    model_paths = {
        "YOLOv10s": "models/YOLOv10s.pt",
        "YOLOv10m": "models/YOLOv10m.pt", 
        "YOLOv11n": "models/YOLOv11n.pt",
        "YOLOv12s": "models/YOLOv12s.pt"
    }
    
    for model_name, path in model_paths.items():
        model_files[model_name] = os.path.exists(path)
    
    return jsonify({
        "status": "healthy" if custom_loaded else "warning",
        "custom_modules_loaded": custom_loaded,
        "model_files": model_files,
        "message": "Custom modules loaded successfully" if custom_loaded else "Custom modules not loaded - YOLOv10+ models may fail"
    })

@app.route("/test", methods=["GET"])
def test_endpoint():
    """Simple test endpoint to verify backend is working"""
    return jsonify({
        "status": "success",
        "message": "Backend is working!",
        "timestamp": time.time(),
        "models_available": ["YOLOv10s", "YOLOv10m", "YOLOv11n", "YOLOv12s"]
    })

@app.route("/test-upload", methods=["POST"])
def test_upload():
    """Test file upload functionality"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Just save and return file info
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"test_{filename}")
        file.save(filepath)
        
        file_size = os.path.getsize(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            "status": "success",
            "message": "File upload test successful",
            "filename": filename,
            "size": file_size,
            "type": file.content_type
        })
        
    except Exception as e:
        return jsonify({"error": f"Upload test failed: {str(e)}"}), 500

@app.route("/memory", methods=["GET"])
def memory_status():
    """Check memory usage and model cache status"""
    try:
        import psutil
        import gc
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        # Get model cache info
        from utils.inference_with_custom import _model_cache
        cache_info = {
            "cached_models": list(_model_cache.keys()),
            "cache_size": len(_model_cache)
        }
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
            "memory_used_mb": memory.used / 1024 / 1024,
            "cache_info": cache_info,
            "status": "healthy" if memory.percent < 80 else "warning"
        })
    except Exception as e:
        return jsonify({"error": f"Memory check failed: {str(e)}"}), 500

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear model cache to free memory"""
    try:
        from utils.inference_with_custom import clear_model_cache
        clear_model_cache()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Cache clear failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)