# ── Dockerfile ──
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements.txt (should include a CUDA-enabled OpenCV wheel and PyTorch with CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your TensorRT engines and inference script
COPY plate.trt ocr.trt helmet.trt inference.py ./

ENTRYPOINT ["python", "inference.py"]
