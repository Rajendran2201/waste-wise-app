import torch
import os
from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image

# Import custom modules first to register them
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import custom_yolo_modules
    from custom_yolo_modules import register_modules, patch_ultralytics_modules
    
    # Register custom modules
    register_modules()
    patch_ultralytics_modules()
    print("✅ Custom YOLO modules loaded and registered")
except ImportError as e:
    print(f"⚠️ Custom modules not found: {e}")
except Exception as e:
    print(f"⚠️ Error loading custom modules: {e}")

# Updated model paths - adjust these to match your actual model locations
MODEL_MAP = {
    "YOLOv10s": "models/YOLOv10s.pt",
    "YOLOv10m": "models/YOLOv10m.pt", 
    "YOLOv11n": "models/YOLOv11n.pt",
    "YOLOv12s": "models/YOLOv12s.pt"
}

def load_model(model_name):
    """Load YOLO model based on model name with fallback methods"""
    model_path = MODEL_MAP.get(model_name)
    if not model_path:
        raise ValueError(f"Model {model_name} not found in MODEL_MAP")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Method 1: Try ultralytics YOLO first
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, "ultralytics"
    except Exception as e1:
        print(f"Ultralytics YOLO failed: {e1}")
        
        try:
            # Method 2: Try torch.hub with custom path
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)
            return model, "torch_hub"
        except Exception as e2:
            print(f"Torch hub failed: {e2}")
            
            try:
                # Method 3: Direct torch load with custom handling
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                model.eval()
                return model, "torch_direct"
            except Exception as e3:
                raise Exception(f"All model loading methods failed. Ultralytics: {e1}, Torch hub: {e2}, Direct: {e3}")

def run_inference(image_path, model_name):
    """Run inference on image and return result path and detection data"""
    try:
        # Load model with fallback methods
        model, method = load_model(model_name)
        
        # Create result directory
        result_dir = Path("static/results")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename for result
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_image_path = result_dir / result_filename
        
        if method == "ultralytics":
            return _run_ultralytics_inference(model, image_path, result_image_path)
        elif method == "torch_hub":
            return _run_torch_hub_inference(model, image_path, result_image_path)
        else:  # torch_direct
            return _run_torch_direct_inference(model, image_path, result_image_path)
        
    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")

def _run_ultralytics_inference(model, image_path, result_image_path):
    """Run inference using ultralytics YOLO"""
    # Run inference
    results = model(image_path)
    
    # Get the first result (assuming single image)
    result = results[0]
    annotated_img = result.plot()
    
    # Save the annotated image
    cv2.imwrite(str(result_image_path), annotated_img)
    
    # Extract detection data
    result_data = []
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            detection = {
                "class_id": int(boxes.cls[i]),
                "class_name": model.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "bbox": boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
            }
            result_data.append(detection)
    
    return str(result_image_path), result_data

def _run_torch_hub_inference(model, image_path, result_image_path):
    """Run inference using torch hub YOLO"""
    # Run inference
    results = model(image_path)
    
    # Load original image for annotation
    img = cv2.imread(image_path)
    
    # Extract detection data and draw bounding boxes
    result_data = []
    
    if hasattr(results, 'pandas'):
        # YOLOv5 style results
        detections = results.pandas().xyxy[0]
        result_data = detections.to_dict(orient="records")
        
        # Draw bounding boxes
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            conf = detection['confidence']
            cls_name = detection['name']
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    else:
        # Handle tensor results directly
        detections = results.xyxy[0].cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            cls_name = model.names[int(cls)] if hasattr(model, 'names') else f"class_{int(cls)}"
            
            result_data.append({
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "confidence": float(conf), "class": int(cls),
                "name": cls_name
            })
            
            # Draw rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save annotated image
    cv2.imwrite(str(result_image_path), img)
    
    return str(result_image_path), result_data

def _run_torch_direct_inference(model, image_path, result_image_path):
    """Run inference using direct torch model"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Resize image to model input size (usually 640x640 for YOLO)
    input_size = 640
    print("img_array type:", type(img_array))
    print("img_array keys (if dict):", img_array.keys() if isinstance(img_array, dict) else "Not a dict")
    h, w = img_array.shape[:2]
    scale = min(input_size/h, input_size/w)
    new_h, new_w = int(h*scale), int(w*scale)
    
    resized_img = cv2.resize(img_array, (new_w, new_h))
    
    # Pad to square
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded_img = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_img
    
    # Convert to tensor
    tensor_img = torch.from_numpy(padded_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Run inference
    with torch.no_grad():
        if hasattr(model, 'forward'):
            results = model.forward(tensor_img)
        else:
            results = model(tensor_img)
    
    # Process results (this is a simplified version)
    result_data = []
    
    # For now, just save the original image with a message
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, "Direct torch inference - results parsing needed", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(result_image_path), img_bgr)
    
    return str(result_image_path), result_data

# Additional utility functions for custom model handling
def download_custom_repo_if_needed():
    """Download custom YOLO repository if needed for special modules"""
    try:
        # You might need to clone specific repositories for YOLOv10/v11/v12
        # Example: git clone https://github.com/THU-MIG/yolov10.git
        pass
    except Exception as e:
        print(f"Failed to download custom repo: {e}")

def get_model_info(model_path):
    """Get information about the model file"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Model keys: {checkpoint.keys()}")
        if 'model' in checkpoint:
            model = checkpoint['model']
            print(f"Model type: {type(model)}")
            if hasattr(model, 'names'):
                print(f"Classes: {model.names}")
        return checkpoint
    except Exception as e:
        print(f"Failed to inspect model: {e}")
        return None