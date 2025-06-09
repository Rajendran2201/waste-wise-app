import torch
import os
from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image
import sys

# Global flag to track if modules are registered
_modules_registered = False

def ensure_custom_modules_global():
    """Ensure custom modules are globally available before any model loading"""
    global _modules_registered
    
    if _modules_registered:
        return True
    
    try:
        # Method 1: Try to import and register custom modules
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import custom modules
        import custom_yolo_modules
        from custom_yolo_modules import register_modules, patch_ultralytics_modules
        
        # Register modules
        register_modules()
        patch_ultralytics_modules()
        
        print("✅ Custom YOLO modules loaded via custom_yolo_modules")
        _modules_registered = True
        return True
        
    except ImportError:
        print("⚠️ custom_yolo_modules not found, trying direct definition...")
        
        # Method 2: Define custom modules directly
        try:
            import torch.nn as nn
            import ultralytics.nn.modules.block as block_module
            
            # Define minimal custom modules that are commonly missing
            class SCDown(nn.Module):
                """Spatial Channel Downsampling - commonly used in YOLOv10+"""
                def __init__(self, c1, c2, k=3, s=2):
                    super().__init__()
                    self.cv1 = nn.Conv2d(c1, c2//2, 1, 1)
                    self.cv2 = nn.Conv2d(c2//2, c2//2, k, s, k//2, groups=c2//2)
                    self.cv3 = nn.Conv2d(c2//2, c2, 1, 1)
                    
                def forward(self, x):
                    return self.cv3(self.cv2(self.cv1(x)))
            
            class TryExcept(nn.Module):
                """Try-except wrapper for modules"""
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                    
                def forward(self, x):
                    try:
                        return self.module(x)
                    except Exception as e:
                        print(f"TryExcept caught: {e}")
                        return x
            
            class C2PSA(nn.Module):
                """C2 with Position-Sensitive Attention"""
                def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
                    super().__init__()
                    self.c = int(c2 * e)
                    self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
                    self.cv2 = nn.Conv2d(2 * self.c, c2, 1)
                    self.m = nn.ModuleList([nn.Conv2d(self.c, self.c, 3, 1, 1, groups=self.c) for _ in range(n)])
                    self.shortcut = shortcut and c1 == c2
                    
                def forward(self, x):
                    y = self.cv1(x)
                    y1, y2 = y.chunk(2, 1)
                    for m in self.m:
                        y1 = m(y1) + y1
                    return self.cv2(torch.cat([y1, y2], 1)) + (x if self.shortcut else 0)
            
            class C2fPSA(nn.Module):
                """C2f with Position-Sensitive Attention"""
                def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
                    super().__init__()
                    self.c = int(c2 * e)
                    self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
                    self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
                    self.m = nn.ModuleList([C2PSA(self.c, self.c, 1, True, g, 1.0) for _ in range(n)])
                    
                def forward(self, x):
                    y = list(self.cv1(x).chunk(2, 1))
                    for m in self.m:
                        y.append(m(y[-1]))
                    return self.cv2(torch.cat(y, 1))
            
            # Register modules globally
            torch.nn.SCDown = SCDown
            torch.nn.TryExcept = TryExcept  
            torch.nn.C2PSA = C2PSA
            torch.nn.C2fPSA = C2fPSA
            
            # Also register in ultralytics if available
            try:
                block_module.SCDown = SCDown
                block_module.TryExcept = TryExcept
                block_module.C2PSA = C2PSA
                block_module.C2fPSA = C2fPSA
                print("✅ Custom modules registered in ultralytics.nn.modules.block")
            except:
                pass
            
            # Register in global namespace for torch.load
            globals()['SCDown'] = SCDown
            globals()['TryExcept'] = TryExcept
            globals()['C2PSA'] = C2PSA
            globals()['C2fPSA'] = C2fPSA
            
            print("✅ Custom modules defined and registered globally")
            _modules_registered = True
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to define custom modules: {e}")
            return False

# Updated model paths
MODEL_MAP = {
    "YOLOv10s": "models/YOLOv10s.pt",
    "YOLOv10m": "models/YOLOv10m.pt", 
    "YOLOv11n": "models/YOLOv11n.pt",
    "YOLOv12s": "models/YOLOv12s.pt"
}

def load_model_safe(model_path):
    """Safely load a YOLO model with custom modules"""
    # Ensure custom modules are available BEFORE loading
    ensure_custom_modules_global()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Try different loading methods in order of preference
    methods = [
        ("ultralytics_yolo", load_with_ultralytics),
        ("torch_direct", load_with_torch_direct),
        ("torch_hub", load_with_torch_hub)
    ]
    
    last_error = None
    for method_name, load_func in methods:
        try:
            print(f"Trying {method_name}...")
            model = load_func(model_path)
            print(f"✅ Successfully loaded model with {method_name}")
            return model, method_name
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            last_error = e
            continue
    
    raise Exception(f"All loading methods failed. Last error: {last_error}")

def load_with_ultralytics(model_path):
    """Load using ultralytics YOLO"""
    from ultralytics import YOLO
    return YOLO(model_path)

def load_with_torch_direct(model_path):
    """Load directly with torch.load"""
    # Set up custom unpickler to handle missing modules
    import pickle
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle custom YOLO modules
            if name in ['SCDown', 'TryExcept', 'C2PSA', 'C2fPSA']:
                return globals().get(name, super().find_class(module, name))
            return super().find_class(module, name)
    
    # Load with custom unpickler
    with open(model_path, 'rb') as f:
        checkpoint = CustomUnpickler(f).load()
    
    if isinstance(checkpoint, dict):
        model = checkpoint.get('model', checkpoint)
    else:
        model = checkpoint
    
    # Ensure model is in eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    return model

def load_with_torch_hub(model_path):
    """Load using torch.hub (fallback)"""
    return torch.hub.load('ultralytics/yolov5', 'custom', 
                         path=model_path, force_reload=True, trust_repo=True)

def run_inference(image_path, model_name):
    """Run inference with improved error handling"""
    try:
        # Ensure custom modules are loaded first
        ensure_custom_modules_global()
        
        # Get model path
        model_path = MODEL_MAP.get(model_name)
        if not model_path:
            raise ValueError(f"Model {model_name} not found in MODEL_MAP")
        
        # Load model
        model, method = load_model_safe(model_path)
        
        # Create result directory
        result_dir = Path("static/results")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_image_path = result_dir / result_filename
        
        # Run inference based on loading method
        if method == "ultralytics_yolo":
            return run_ultralytics_inference(model, image_path, result_image_path)
        elif method == "torch_hub":
            return run_torch_hub_inference(model, image_path, result_image_path)
        else:  # torch_direct
            return run_torch_direct_inference(model, image_path, result_image_path)
            
    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")

def run_ultralytics_inference(model, image_path, result_image_path):
    """Run inference using ultralytics YOLO"""
    results = model(image_path)
    result = results[0]
    
    # Save annotated image
    annotated_img = result.plot()
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
                "bbox": boxes.xyxy[i].tolist()
            }
            result_data.append(detection)
    
    return str(result_image_path), result_data

def run_torch_hub_inference(model, image_path, result_image_path):
    """Run inference using torch hub"""
    results = model(image_path)
    img = cv2.imread(image_path)
    result_data = []
    
    if hasattr(results, 'pandas'):
        detections = results.pandas().xyxy[0]
        result_data = detections.to_dict(orient="records")
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            conf = detection['confidence']
            cls_name = detection['name']
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(result_image_path), img)
    return str(result_image_path), result_data

def run_torch_direct_inference(model, image_path, result_image_path):
    """Run inference using direct torch model"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Simple preprocessing for YOLO input
    input_size = 640
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
    
    # Basic result processing (you may need to customize this)
    result_data = []
    
    # Save image with basic annotation
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, f"Inference completed - {len(results) if isinstance(results, (list, tuple)) else 'unknown'} outputs", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(result_image_path), img_bgr)
    
    return str(result_image_path), result_data

# Initialize custom modules when this module is imported
ensure_custom_modules_global()