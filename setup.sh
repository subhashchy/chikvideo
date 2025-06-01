#!/bin/bash

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p media uploads

echo "Setup completed! You can now run: streamlit run face_recognition.py" 