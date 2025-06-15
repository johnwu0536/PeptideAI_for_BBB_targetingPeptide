#!/bin/bash

# Script to run PeptideAI in CPU-only mode
echo "PeptideAI - CPU-Only Mode"
echo "========================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install conda first: https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi

# Create and activate a new conda environment
echo "Creating a new conda environment for CPU-only mode..."
conda create -y -n peptideai_cpu python=3.9
eval "$(conda shell.bash hook)"
conda activate peptideai_cpu

# Install PyTorch without CUDA support
echo "Installing PyTorch without CUDA support..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install torchdata
echo "Installing torchdata..."
pip install torchdata==0.7.0

# Install PyTorch Geometric without CUDA support
echo "Installing PyTorch Geometric without CUDA support..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Check for GLIBCXX_3.4.29
echo "Checking for GLIBCXX_3.4.29..."
if ! strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29 > /dev/null; then
    echo "Warning: GLIBCXX_3.4.29 not found in system libstdc++.so.6"
    echo "Installing scipy<1.10.0 to avoid compatibility issues..."
    pip install "scipy<1.10.0"
fi

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import torchdata; print(f'torchdata version: {torchdata.__version__}'); import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"

# Run the test installation script
echo "Running comprehensive installation test..."
python test_installation.py

# Run the model in CPU mode
echo
echo "You can now run the model in CPU-only mode with:"
echo "conda activate peptideai_cpu"
echo "python main.py --mode train --data_path data/Peptide.csv --device cpu"
echo
echo "Note: Training will be much slower in CPU-only mode."
echo "If you want to use GPU acceleration, please fix the CUDA libraries issue with:"
echo "sudo ./fix_cuda_libs.sh"
