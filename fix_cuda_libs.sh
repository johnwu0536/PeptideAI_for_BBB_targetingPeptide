#!/bin/bash

# Script to fix missing CUDA libraries for PeptideAI
echo "PeptideAI - CUDA Libraries Fix"
echo "============================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

echo "This script will install missing CUDA libraries required by PyTorch and PyTorch Geometric."
echo "It will download and install the necessary libraries from the NVIDIA repository."
echo

# Add NVIDIA repository
echo "Adding NVIDIA repository..."
apt-get update
apt-get install -y software-properties-common
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update

# Install CUDA libraries
echo "Installing CUDA libraries..."
apt-get install -y --no-install-recommends \
    libcusparse-12-0 \
    libcublas-12-0 \
    libcufft-12-0 \
    libcurand-12-0 \
    libcusolver-12-0 \
    libcudnn8

# Create symbolic links if needed
echo "Creating symbolic links..."
if [ ! -f /usr/lib/x86_64-linux-gnu/libcusparse.so.12 ]; then
    ln -s /usr/local/cuda-12.*/lib64/libcusparse.so.12 /usr/lib/x86_64-linux-gnu/libcusparse.so.12
fi

if [ ! -f /usr/lib/x86_64-linux-gnu/libcublas.so.12 ]; then
    ln -s /usr/local/cuda-12.*/lib64/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.12
fi

# Alternative approach: Install the full CUDA toolkit
echo
echo "If the above steps didn't work, you can install the full CUDA toolkit:"
echo "sudo apt-get install -y cuda-toolkit-12-1"
echo

echo "Done! You should now be able to run PeptideAI with CUDA support."
echo "If you still encounter issues, please try installing the full CUDA toolkit."
