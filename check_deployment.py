#!/usr/bin/env python3
"""
Deployment Check Script for DetectoReal
This script verifies that all necessary files are present for Streamlit Cloud deployment.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        print(f"✅ {description}: {filepath} ({size:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def main():
    print("🔍 Checking DetectoReal deployment files...\n")
    
    # List of required files
    required_files = [
        ("app.py", "Main Streamlit application"),
        ("predict.py", "Prediction module"),
        ("model.py", "Model architecture"),
        ("requirements.txt", "Python dependencies"),
        ("model.pth", "PyTorch model weights"),
    ]
    
    # List of optional model files
    optional_files = [
        ("real_vs_fake.keras", "Keras model (alternative)"),
        ("real_vs_fake.h5", "H5 model (alternative)"),
    ]
    
    print("📋 Required files:")
    all_required_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_required_present = False
    
    print("\n📋 Optional model files:")
    for filepath, description in optional_files:
        check_file_exists(filepath, description)
    
    print("\n🔧 Deployment recommendations:")
    
    if all_required_present:
        print("✅ All required files are present!")
        print("✅ Ready for Streamlit Cloud deployment")
    else:
        print("❌ Some required files are missing!")
        print("❌ Please ensure all required files are present before deployment")
    
    print("\n📝 Next steps:")
    print("1. Commit all files to your repository")
    print("2. Push to GitHub/GitLab")
    print("3. Deploy to Streamlit Cloud")
    print("4. The app should now work without FileNotFoundError")

if __name__ == "__main__":
    main() 