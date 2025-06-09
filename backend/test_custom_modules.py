#!/usr/bin/env python3
"""
Test script to verify custom YOLO modules are properly loaded
"""

import sys
import os

def test_custom_modules():
    """Test if custom modules can be imported and registered"""
    print("🧪 Testing custom YOLO modules...")
    
    try:
        # Add backend directory to path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, backend_dir)
        
        # Import custom modules
        import custom_yolo_modules
        print("✅ Custom modules imported successfully")
        
        # Test if SCDown is available
        from custom_yolo_modules import SCDown
        print("✅ SCDown module found")
        
        # Test if TryExcept is available
        from custom_yolo_modules import TryExcept
        print("✅ TryExcept module found")
        
        # Register modules
        from custom_yolo_modules import register_modules, patch_ultralytics_modules
        register_modules()
        patch_ultralytics_modules()
        print("✅ Modules registered successfully")
        
        # Test ultralytics import
        try:
            from ultralytics import YOLO
            print("✅ Ultralytics YOLO imported successfully")
        except Exception as e:
            print(f"⚠️ Ultralytics import issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing custom modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Check if models exist
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(models_dir):
            print(f"❌ Models directory not found: {models_dir}")
            return False
            
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        print(f"📁 Found {len(model_files)} model files: {model_files}")
        
        if not model_files:
            print("❌ No model files found")
            return False
            
        # Test loading first available model
        model_path = os.path.join(models_dir, model_files[0])
        print(f"🔍 Testing model: {model_path}")
        
        # Import inference module
        from utils.inference_with_custom import load_model
        
        # Get model name from filename
        model_name = model_files[0].replace('.pt', '')
        print(f"📋 Model name: {model_name}")
        
        # Try to load model
        model, method = load_model(model_name)
        print(f"✅ Model loaded successfully using method: {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Custom YOLO Modules Test")
    print("=" * 50)
    
    # Test custom modules
    modules_ok = test_custom_modules()
    
    # Test model loading
    models_ok = test_model_loading()
    
    print("\n" + "=" * 50)
    if modules_ok and models_ok:
        print("✅ All tests passed! Your setup should work.")
    else:
        print("❌ Some tests failed. Check the errors above.") 