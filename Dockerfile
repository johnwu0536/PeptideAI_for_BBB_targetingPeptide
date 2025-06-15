# Dockerfile for PeptideAI
# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir torchdata==0.7.0
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Make scripts executable
RUN chmod +x setup.sh fix_libstdcpp.sh

# Verify installation
RUN python test_installation.py

# Set default command
CMD ["python", "main.py", "--mode", "train", "--data_path", "data/Peptide.csv", "--device", "cuda"]
