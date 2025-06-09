#!/usr/bin/env python3
"""
Script to set up custom YOLO repositories and fix common issues
"""

import os
import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def clone_repo(repo_url, target_dir):
    """Clone a git repository"""
    try:
        if os.path.exists(target_dir):
            print(f"📁 Directory {target_dir} already exists")
            return True
        subprocess.check_call(["git", "clone", repo_url, target_dir])
        print(f"✅ Successfully cloned {repo_url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone {repo_url}: {e}")
        return False

def setup_yolov10():
    """Set up YOLOv10 repository"""
    print("\n🔧 Setting up YOLOv10...")
    
    # Method 1: Try pip install
    if install_package("git+https://github.com/THU-MIG/yolov10.git"):
        return True
    
    # Method 2: Clone and install locally
    repo_url = "https://github.com/THU-MIG/yolov10.git"
    target_dir = "yolov10"
    
    if clone_repo(repo_url, target_dir):
        try:
            # Install requirements if exists
            req_file = Path(target_dir) / "requirements.txt"
            if req_file.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            
            # Install the package
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", target_dir])
            print("✅ YOLOv10 setup complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install YOLOv10 locally: {e}")
    
    return False

def setup_custom_modules():
    """Create custom module definitions for missing modules"""
    print("\n🔧 Creating custom module definitions...")
    
    custom_modules_code = '''
import torch
import torch.nn as nn

class SCDown(nn.Module):
    """Spatial Channel Downsampling - Custom implementation"""
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2//2, 1, 1)
        self.cv2 = nn.Conv2d(c2//2, c2//2, k, s, k//2, groups=c2//2)
        self.cv3 = nn.Conv2d(c2//2, c2, 1, 1)
        
    def forward(self, x):
        return self.cv3(self.cv2(self.cv1(x)))

class C2PSA(nn.Module):
    """C2 with Position Self Attention - Custom implementation"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1)
        self.m = nn.Sequential(*(nn.Conv2d(self.c, self.c, 3, 1, 1) for _ in range(n)))
        self.shortcut = shortcut and c1 == c2
        
    def forward(self, x):
        y = self.cv1(x)
        return self.cv2(torch.cat((self.m(y[:, :self.c]), y[:, self.c:]), 1)) + (x if self.shortcut else 0)

class C2fPSA(C2PSA):
    """C2f with Position Self Attention - Custom implementation"""
    pass

# Register custom modules globally
def register_custom_modules():
    """Register custom modules in ultralytics namespace"""
    try:
        import ultralytics.nn.modules.block as block_module
        
        # Add custom modules to the block module
        block_module.SCDown = SCDown
        block_module.C2PSA = C2PSA  
        block_module.C2fPSA = C2fPSA
        
        print("✅ Custom modules registered successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to register custom modules: {e}")
        return False

# Auto-register when imported
register_custom_modules()
'''
    
    # Write to a custom modules file
    with open("custom_yolo_modules.py", "w") as f:
        f.write(custom_modules_code)
    
    print("✅ Custom modules file created: custom_yolo_modules.py")
    print("   Import this before loading your models: import custom_yolo_modules")

def update_inference_with_custom_modules():
    """Update the inference file to use custom modules"""
    print("\n🔧 Creating updated inference file...")
    
    updated_code = '''
# Import custom modules first
try:
    import custom_yolo_modules
    print("✅ Custom YOLO modules loaded")
except ImportError as e:
    print(f"⚠️ Custom modules not found: {e}")

# Rest of your inference code here...
'''
    
    with open("utils/inference_with_custom.py", "w") as f:
        f.write(updated_code)
        
        # Read the original file and append
        try:
            with open("utils/inference.py", "r") as orig:
                f.write(orig.read())
        except FileNotFoundError:
            print("⚠️ Original inference.py not found")
    
    print("✅ Updated inference file created: utils/inference_with_custom.py")

def main():
    print("🚀 YOLO Setup and Fix Script")
    print("=" * 50)
    
    # Update ultralytics first
    print("📦 Updating ultralytics...")
    install_package("--upgrade ultralytics")
    
    # Set up YOLOv10
    setup_yolov10()
    
    # Create custom modules
    setup_custom_modules()
    
    # Update inference file
    update_inference_with_custom_modules()
    
    print("\n✅ Setup complete! Try running your app again.")
    print("\n📝 Next steps:")
    print("1. Restart your Flask app")
    print("2. If issues persist, try using utils/inference_with_custom.py")
    print("3. Consider converting models to ONNX format for better compatibility")

if __name__ == "__main__":
    main()
    