#!/bin/bash

# Setup script for PeptideAI project
echo "Setting up PeptideAI project..."

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "py39"; then
    echo "Creating conda environment 'py39'..."
    conda create -y -n py39 python=3.9
fi

# Activate conda environment
echo "Activating conda environment 'py39'..."
eval "$(conda shell.bash hook)"
conda activate py39

# Install PyTorch with CUDA 12.6 support
echo "Installing PyTorch with CUDA 12.6 support..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install torchdata explicitly first
echo "Installing torchdata..."
pip install torchdata==0.7.0

# Install PyTorch Geometric with CUDA support
echo "Installing PyTorch Geometric with CUDA support..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Check for GLIBCXX_3.4.29
echo "Checking for GLIBCXX_3.4.29..."
if ! strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29 > /dev/null; then
    echo "Warning: GLIBCXX_3.4.29 not found in system libstdc++.so.6"
    echo "Using older versions of dependencies to avoid compatibility issues"
    
    # Install specific versions of dependencies that don't require GLIBCXX_3.4.29
    echo "Installing scipy<1.10.0 to avoid GLIBCXX_3.4.29 dependency..."
    pip install "scipy<1.10.0"
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Check for CUDA libraries
echo "Checking for CUDA libraries..."
if [ -f /usr/lib/x86_64-linux-gnu/libcusparse.so.12 ]; then
    echo "libcusparse.so.12 found."
else
    echo "Warning: libcusparse.so.12 not found."
    echo "This library is required for PyTorch with CUDA support."
    echo "Please run the fix_cuda_libs.sh script as root to install the missing libraries:"
    echo "sudo ./fix_cuda_libs.sh"
    echo "Alternatively, you can install the full CUDA toolkit:"
    echo "sudo apt-get install -y cuda-toolkit-12-1"
    echo "Continuing with installation, but CUDA support may not work correctly."
fi

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); import torchdata; print(f'torchdata version: {torchdata.__version__}'); import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"

# Run the test installation script
echo "Running comprehensive installation test..."
python test_installation.py

# Check for errors during installation test
if [ $? -ne 0 ]; then
    echo "Installation test failed. This may be due to missing CUDA libraries."
    echo "Please run the fix_cuda_libs.sh script as root to install the missing libraries:"
    echo "sudo ./fix_cuda_libs.sh"
fi

echo "Setup complete! You can now run the project with:"
echo "python main.py --mode train --data_path data/Peptide.csv --device cuda"
