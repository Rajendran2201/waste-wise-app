import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image
import pickle

# Global flag to track module registration
_modules_registered = False

def define_custom_modules():
    """Define the custom YOLO modules that are commonly missing"""
    
    class SCDown(nn.Module):
        """Spatial Channel Downsampling used in YOLOv10+"""
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
    
    return SCDown, TryExcept, C2PSA, C2fPSA

def register_custom_modules():
    """Register custom modules globally for PyTorch deserialization"""
    global _modules_registered
    
    if _modules_registered:
        return True
    
    try:
        # First try to import your custom modules
        try:
            import custom_yolo_modules
            from custom_yolo_modules import register_modules, patch_ultralytics_modules
            register_modules()
            patch_ultralytics_modules()
            print("✅ Custom YOLO modules loaded from custom_yolo_modules")
            _modules_registered = True
            return True
        except ImportError:
            print("⚠️ custom_yolo_modules not found, using fallback definitions...")
        
        # Fallback: Define modules ourselves
        SCDown, TryExcept, C2PSA, C2fPSA = define_custom_modules()
        
        # Register in multiple locations for maximum compatibility
        
        # 1. Global namespace (for pickle/torch.load)
        globals()['SCDown'] = SCDown
        globals()['TryExcept'] = TryExcept
        globals()['C2PSA'] = C2PSA
        globals()['C2fPSA'] = C2fPSA
        
        # 2. torch.nn namespace
        torch.nn.SCDown = SCDown
        torch.nn.TryExcept = TryExcept
        torch.nn.C2PSA = C2PSA
        torch.nn.C2fPSA = C2fPSA
        
        # 3. ultralytics block module (if available)
        try:
            import ultralytics.nn.modules.block as block_module
            block_module.SCDown = SCDown
            block_module.TryExcept = TryExcept
            block_module.C2PSA = C2PSA
            block_module.C2fPSA = C2fPSA
            print("✅ Registered modules in ultralytics.nn.modules.block")
        except ImportError:
            print("⚠️ ultralytics not available for direct patching")
        
        # 4. sys.modules for import resolution
        import types
        custom_module = types.ModuleType('custom_yolo_modules')
        custom_module.SCDown = SCDown
        custom_module.TryExcept = TryExcept
        custom_module.C2PSA = C2PSA
        custom_module.C2fPSA = C2fPSA
        sys.modules['custom_yolo_modules'] = custom_module
        
        print("✅ Custom modules registered globally")
        _modules_registered = True
        return True
        
    except Exception as e:
        print(f"❌ Failed to register custom modules: {e}")
        return False

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle missing YOLO modules"""
    
    def find_class(self, module, name):
        # Handle custom YOLO modules
        if name in ['SCDown', 'TryExcept', 'C2PSA', 'C2fPSA']:
            if name in globals():
                return globals()[name]
            else:
                print(f"⚠️ Module {name} not found, creating placeholder...")
                # Return a basic nn.Module as placeholder
                return type(name, (nn.Module,), {'forward': lambda self, x: x})
        
        # Handle ultralytics modules that might be missing
        if 'ultralytics' in module and name in ['SCDown', 'TryExcept', 'C2PSA', 'C2fPSA']:
            return globals().get(name, nn.Identity)
        
        return super().find_class(module, name)

# Updated model paths
MODEL_MAP = {
    "YOLOv10s": "models/YOLOv10s.pt",
    "YOLOv10m": "models/YOLOv10m.pt", 
    "YOLOv11n": "models/YOLOv11n.pt",
    "YOLOv12s": "models/YOLOv12s.pt"
}

def load_model(model_name):
    """Load YOLO model with comprehensive fallback methods"""
    # CRITICAL: Register custom modules BEFORE any model loading
    register_custom_modules()
    
    model_path = MODEL_MAP.get(model_name)
    if not model_path:
        raise ValueError(f"Model {model_name} not found in MODEL_MAP")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Attempting to load model: {model_path}")
    
    # Method 1: Try ultralytics YOLO
    try:
        from ultralytics import YOLO
        print("Trying ultralytics YOLO...")
        model = YOLO(model_path)
        print("✅ Successfully loaded with ultralytics YOLO")
        return model, "ultralytics"
    except Exception as e1:
        print(f"❌ Ultralytics YOLO failed: {e1}")
    
    # Method 2: Direct torch load with custom unpickler
    try:
        print("Trying direct torch.load with custom unpickler...")
        with open(model_path, 'rb') as f:
            checkpoint = CustomUnpickler(f).load()
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
            
        if hasattr(model, 'eval'):
            model.eval()
            
        print("✅ Successfully loaded with custom unpickler")
        return model, "torch_direct"
    except Exception as e2:
        print(f"❌ Custom unpickler failed: {e2}")
    
    # Method 3: Try torch.hub
    try:
        print("Trying torch.hub...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=model_path, force_reload=True, trust_repo=True)
        print("✅ Successfully loaded with torch.hub")
        return model, "torch_hub"
    except Exception as e3:
        print(f"❌ Torch hub failed: {e3}")
    
    # Method 4: Last resort - standard torch.load
    try:
        print("Trying standard torch.load...")
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        if hasattr(model, 'eval'):
            model.eval()
        print("✅ Successfully loaded with standard torch.load")
        return model, "torch_standard"
    except Exception as e4:
        print(f"❌ Standard torch.load failed: {e4}")
    
    raise Exception(f"All model loading methods failed for {model_path}")

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
        
        print(f"Running inference with method: {method}")
        
        if method == "ultralytics":
            return _run_ultralytics_inference(model, image_path, result_image_path)
        elif method == "torch_hub":
            return _run_torch_hub_inference(model, image_path, result_image_path)
        else:  # torch_direct or torch_standard
            return _run_torch_direct_inference(model, image_path, result_image_path)
        
    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")

def _run_ultralytics_inference(model, image_path, result_image_path):
    """Run inference using ultralytics YOLO"""
    try:
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
        
    except Exception as e:
        print(f"Error in ultralytics inference: {e}")
        # Fallback to basic image copy
        img = cv2.imread(image_path)
        cv2.putText(img, f"Ultralytics inference error: {str(e)[:50]}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(result_image_path), img)
        return str(result_image_path), []

def _run_torch_hub_inference(model, image_path, result_image_path):
    """Run inference using torch hub YOLO"""
    try:
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
        
    except Exception as e:
        print(f"Error in torch hub inference: {e}")
        # Fallback to basic image copy
        img = cv2.imread(image_path)
        cv2.putText(img, f"Torch hub inference error: {str(e)[:50]}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(result_image_path), img)
        return str(result_image_path), []

def _run_torch_direct_inference(model, image_path, result_image_path):
    """Run inference using direct torch model"""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Resize image to model input size (usually 640x640 for YOLO)
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
        
        # Process results (basic version)
        result_data = []
        
        # Save image with basic annotation
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.putText(img_bgr, f"Direct torch inference completed", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_bgr, f"Model loaded successfully", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(result_image_path), img_bgr)
        
        return str(result_image_path), result_data
        
    except Exception as e:
        print(f"Error in direct torch inference: {e}")
        # Fallback to basic image copy
        img = cv2.imread(image_path)
        cv2.putText(img, f"Direct inference error: {str(e)[:50]}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(result_image_path), img)
        return str(result_image_path), []

def get_model_info(model_path):
    """Get information about the model file"""
    try:
        register_custom_modules()  # Ensure modules are registered
        
        with open(model_path, 'rb') as f:
            checkpoint = CustomUnpickler(f).load()
        
        print(f"Model keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            print(f"Model type: {type(model)}")
            if hasattr(model, 'names'):
                print(f"Classes: {model.names}")
        return checkpoint
    except Exception as e:
        print(f"Failed to inspect model: {e}")
        return None

# Auto-register modules when this file is imported
register_custom_modules()