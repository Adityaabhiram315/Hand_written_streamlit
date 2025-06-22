#!/usr/bin/env python3
"""
Quick verification script to check if all dependencies are properly installed
"""

import sys

def check_imports():
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import streamlit as st
        print(f"✅ Streamlit: {st.__version__}")
    except ImportError:
        print("❌ Streamlit not found")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    try:
        import PIL
        print(f"✅ Pillow: {PIL.__version__}")
    except ImportError:
        print("❌ Pillow not found")
        return False
    
    return True

if __name__ == "__main__":
    print("Checking MNIST project dependencies...")
    print("=" * 40)
    
    if check_imports():
        print("\n🎉 All dependencies are properly installed!")
        print("You can now run: streamlit run Streamlit_app.py")
    else:
        print("\n❌ Some dependencies are missing.")
        print("Please run: pip install -r requirements.txt")
