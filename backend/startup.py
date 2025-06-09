#!/usr/bin/env python3
"""
Startup script to ensure custom YOLO modules are properly registered
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_custom_modules():
    """Set up custom YOLO modules before any other imports"""
    try:
        # Add backend directory to path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, backend_dir)
        
        # Import and register custom modules
        import custom_yolo_modules
        from custom_yolo_modules import register_modules, patch_ultralytics_modules
        
        # Register custom modules with ultralytics
        register_modules()
        patch_ultralytics_modules()
        
        logger.info("✅ Custom YOLO modules loaded and registered successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import custom_yolo_modules: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to register custom modules: {e}")
        return False

def patch_ultralytics_directly():
    """Directly patch ultralytics modules"""
    try:
        import ultralytics.nn.modules.block as block_module
        
        # Import custom modules
        from custom_yolo_modules import SCDown, TryExcept, C2PSA, C2fPSA
        
        # Directly add to ultralytics block module
        block_module.SCDown = SCDown
        block_module.TryExcept = TryExcept
        block_module.C2PSA = C2PSA
        block_module.C2fPSA = C2fPSA
        
        logger.info("✅ Direct ultralytics patching successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct patching failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up custom YOLO modules...")
    
    # Set up custom modules
    setup_ok = setup_custom_modules()
    
    # Try direct patching as fallback
    if not setup_ok:
        print("🔄 Trying direct patching...")
        setup_ok = patch_ultralytics_directly()
    
    if setup_ok:
        print("✅ Custom modules setup complete!")
    else:
        print("❌ Custom modules setup failed!")
        sys.exit(1) 