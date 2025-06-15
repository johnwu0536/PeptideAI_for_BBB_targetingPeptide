# Running PeptideAI with Docker

This document provides instructions for running PeptideAI using Docker, which can help avoid dependency issues like the GLIBCXX_3.4.29 error.

## Prerequisites

- Docker installed on your system
- NVIDIA Container Toolkit (for GPU support)
- Docker Compose (optional, but recommended)

## Option 1: Using Docker Compose (Recommended)

Docker Compose simplifies the process of building and running the container.

### Step 1: Build and Run the Container

```bash
# Navigate to the PeptideAI directory
cd PeptideAI

# Build and run the container in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 2: Execute Commands in the Container

```bash
# Run the training process
docker-compose exec peptideai python main.py --mode train --data_path data/Peptide.csv --device cuda

# Or open a bash shell for interactive use
docker-compose exec peptideai bash
```

### Step 3: Stop the Container

```bash
docker-compose down
```

## Option 2: Using Docker Directly

If you prefer not to use Docker Compose, you can use Docker commands directly.

### Step 1: Build the Docker Image

```bash
# Navigate to the PeptideAI directory
cd PeptideAI

# Build the Docker image
docker build -t peptideai .
```

### Step 2: Run the Container

```bash
# Run the container with GPU support
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  peptideai

# For interactive use (bash shell)
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  peptideai bash
```

### Step 3: Execute Commands in the Container

If you started the container in interactive mode, you can run commands directly:

```bash
# Run the training process
python main.py --mode train --data_path data/Peptide.csv --device cuda

# Test the installation
python test_installation.py

# Run the PyG implementation test
python test_pyg_implementation.py
```

## Customizing the Docker Setup

### Modifying the Dockerfile

You can customize the Dockerfile to suit your needs:

1. Change the base image if you need a different CUDA version
2. Add additional system dependencies
3. Modify the Python package installation process

### Customizing docker-compose.yml

You can customize the docker-compose.yml file:

1. Change the volume mounts to include additional directories
2. Modify the GPU resource allocation
3. Change the default command

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related issues, make sure:

1. The NVIDIA Container Toolkit is installed correctly
2. Your GPU drivers are compatible with the CUDA version in the container
3. The `--gpus all` flag is included when running the container

### Permission Issues

If you encounter permission issues with the mounted volumes:

```bash
# Fix permissions for the outputs directory
chmod -R 777 outputs/
```

### Container Crashes

If the container crashes, check the logs:

```bash
docker logs peptideai
```

## Benefits of Using Docker

Using Docker for PeptideAI provides several advantages:

1. **Consistent Environment**: The same environment is used regardless of the host system
2. **Dependency Isolation**: All dependencies are contained within the Docker image
3. **Avoid System Conflicts**: No need to worry about system library versions (like libstdc++)
4. **Easy Deployment**: The same Docker image can be used on different machines
5. **Reproducibility**: Ensures consistent results across different environments

## Performance Considerations

When running PeptideAI in Docker:

1. **GPU Passthrough**: Docker uses NVIDIA Container Toolkit to pass through the GPU
2. **Volume Mounts**: Using volume mounts ensures data persistence
3. **Resource Allocation**: You can limit CPU and memory usage if needed

For optimal performance, ensure your host system has sufficient resources (CPU, RAM, GPU memory) for the workload.
