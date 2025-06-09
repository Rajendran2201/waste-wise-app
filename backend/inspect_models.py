#!/usr/bin/env python3
"""
Script to inspect YOLO model files and help debug loading issues
"""

import torch
import os
from pathlib import Path

MODEL_MAP = {
    "YOLOv10s": "models/YOLOv10s.pt",
    "YOLOv10m": "models/YOLOv10m.pt", 
    "YOLOv11n": "models/YOLOv11n.pt",
    "YOLOv12s": "models/YOLOv12s.pt"
}

def inspect_model(model_path, model_name):
    """Inspect a YOLO model file"""
    print(f"\n{'='*50}")
    print(f"Inspecting {model_name}: {model_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check model structure
        if 'model' in checkpoint:
            model = checkpoint['model']
            print(f"🔍 Model type: {type(model)}")
            
            if hasattr(model, 'names'):
                print(f"🏷️  Classes: {model.names}")
            
            if hasattr(model, 'yaml'):
                print(f"📋 Model config: {model.yaml}")
                
            # Check for custom modules
            if hasattr(model, 'model'):
                modules = []
                for name, module in model.model.named_modules():
                    module_type = type(module).__name__
                    if module_type not in ['Sequential', 'ModuleList', 'Identity']:
                        modules.append(f"{name}: {module_type}")
                
                print(f"🧩 Model modules:")
                for module in modules[:10]:  # Show first 10
                    print(f"   - {module}")
                if len(modules) > 10:
                    print(f"   ... and {len(modules) - 10} more")
        
        # Check for problematic modules
        model_str = str(checkpoint)
        problematic_modules = ['SCDown', 'C2PSA', 'C2fPSA', 'SCDown']
        found_issues = []
        
        for module in problematic_modules:
            if module in model_str:
                found_issues.append(module)
        
        if found_issues:
            print(f"⚠️  Found potentially problematic modules: {found_issues}")
            print("   These modules might not be available in standard ultralytics")
        else:
            print("✅ No obvious problematic modules detected")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Error type: {type(e).__name__}")

def test_loading_methods(model_path, model_name):
    """Test different loading methods"""
    print(f"\n🧪 Testing loading methods for {model_name}")
    print("-" * 40)
    
    # Method 1: Ultralytics YOLO
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Ultralytics YOLO: SUCCESS")
    except Exception as e:
        print(f"❌ Ultralytics YOLO: FAILED - {e}")
    
    # Method 2: Torch hub
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)
        print("✅ Torch hub: SUCCESS")
    except Exception as e:
        print(f"❌ Torch hub: FAILED - {e}")
    
    # Method 3: Direct torch load
    try:
        model = torch.load(model_path, map_location='cpu')
        print("✅ Direct torch load: SUCCESS")
    except Exception as e:
        print(f"❌ Direct torch load: FAILED - {e}")

def suggest_fixes(model_name):
    """Suggest potential fixes based on model name"""
    print(f"\n💡 Suggestions for {model_name}:")
    
    if "YOLOv10" in model_name:
        print("   - YOLOv10 requires custom repository: https://github.com/THU-MIG/yolov10")
        print("   - Install: pip install git+https://github.com/THU-MIG/yolov10.git")
        print("   - Or download the repo and add to Python path")
    
    elif "YOLOv11" in model_name:
        print("   - Make sure you have the latest ultralytics version")
        print("   - Try: pip install --upgrade ultralytics")
    
    elif "YOLOv12" in model_name:
        print("   - YOLOv12 might need specific ultralytics version or custom repo")
        print("   - Check the source where you got the model for requirements")
    
    print("   - Alternative: Convert model to ONNX format for better compatibility")
    print("   - Or retrain with standard ultralytics YOLO architecture")

def main():
    print("🔍 YOLO Model Inspector")
    print("=" * 50)
    
    for model_name, model_path in MODEL_MAP.items():
        inspect_model(model_path, model_name)
        test_loading_methods(model_path, model_name)
        suggest_fixes(model_name)
        print("\n" + "="*70)

if __name__ == "__main__":
    main()