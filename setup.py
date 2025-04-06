#!/usr/bin/env python3
"""
Setup script for Symptom Analyzer project
Installs dependencies and checks environment
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    major, minor, _ = platform.python_version_tuple()
    major, minor = int(major), int(minor)
    
    if major < 3 or (major == 3 and minor < 6):
        print(f"Error: Python 3.6+ required, but you have {platform.python_version()}")
        return False
    
    print(f"✓ Python version is compatible: {platform.python_version()}")
    return True

def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    # Check if virtual environment is active
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("Warning: It's recommended to install dependencies in a virtual environment.")
        print("You can create one with: python -m venv env")
        print("Then activate it and run this script again.")
        
        proceed = input("Continue with installation anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Installation aborted.")
            return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def check_tensorflow():
    """Check if TensorFlow is available and configured properly"""
    print("\nChecking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version {tf.__version__} is installed")
        
        # Check if GPU is available
        if tf.config.list_physical_devices('GPU'):
            print("✓ TensorFlow is configured with GPU support")
        else:
            print("Note: TensorFlow is running on CPU only (which is fine for this project)")
        
        return True
    except ImportError:
        print("Note: TensorFlow is not installed. Deep learning model will be unavailable.")
        print("To install TensorFlow, run: pip install tensorflow")
        return False
    except Exception as e:
        print(f"Warning: TensorFlow is installed but there might be issues: {str(e)}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("✓ Project directories created")
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("SYMPTOM ANALYZER SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Check TensorFlow
    check_tensorflow()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the training program:")
    print("  python predict_disease.py")
    print("\nAfter training, use the prediction tool:")
    print("  python predict_symptoms.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 