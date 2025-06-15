#!/bin/bash

# Script to fix libstdc++ compatibility issues for PeptideAI
echo "PeptideAI - libstdc++ Compatibility Fix"
echo "========================================"

# Check if GLIBCXX_3.4.29 is available
echo "Checking for GLIBCXX_3.4.29..."
if strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29 > /dev/null; then
    echo "GLIBCXX_3.4.29 is available. No fix needed."
    exit 0
fi

echo "GLIBCXX_3.4.29 not found in system libstdc++.so.6"
echo "This script will help you fix the compatibility issue."
echo

# Option 1: Install older scipy version
echo "Option 1: Install older scipy version (Recommended)"
echo "This option will install scipy<1.10.0 which doesn't require GLIBCXX_3.4.29"
echo "Command: pip install \"scipy<1.10.0\""
echo

# Option 2: Use conda to install a compatible environment
echo "Option 2: Use conda to create a compatible environment"
echo "This option will create a new conda environment with all dependencies"
echo "Command: conda create -n peptideai_env -c conda-forge python=3.9 scipy=1.9.3 numpy pandas scikit-learn matplotlib seaborn tqdm networkx pyyaml"
echo "Then: conda activate peptideai_env"
echo "Then: ./setup.sh"
echo

# Option 3: Use a container
echo "Option 3: Use a container (Advanced)"
echo "This option will use Docker to create a container with all dependencies"
echo "1. Install Docker if not already installed"
echo "2. Create a Dockerfile with the following content:"
echo "   FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
echo "   WORKDIR /app"
echo "   COPY . /app"
echo "   RUN apt-get update && apt-get install -y build-essential"
echo "   RUN pip install -r requirements.txt"
echo "   RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html"
echo "3. Build the Docker image: docker build -t peptideai ."
echo "4. Run the container: docker run --gpus all -it peptideai bash"
echo

# Implement Option 1 automatically
read -p "Do you want to implement Option 1 (install older scipy version)? (y/n): " choice
if [[ $choice == "y" || $choice == "Y" ]]; then
    echo "Installing scipy<1.10.0..."
    pip uninstall -y scipy
    pip install "scipy<1.10.0"
    echo "Done! You should now be able to run PeptideAI without the GLIBCXX_3.4.29 error."
else
    echo "No changes made. Please implement one of the options manually."
fi

echo
echo "For more information, see the README.md file."
