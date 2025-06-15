# PeptideAI

An explainable AI model for peptide binding energy prediction using multimodal data fusion, cross-modal attention layers, graph neural networks, and transformer architecture.

## Overview

PeptideAI is a comprehensive framework for predicting peptide binding energies with a focus on explainability. The model integrates multiple data modalities (peptide sequences, binding energies, and physicochemical properties) using a sophisticated architecture that combines transformers, graph neural networks, and cross-modal attention mechanisms.

Key features:
- Multimodal data fusion with cross-modal attention layers
- Graph neural networks for capturing peptide structural information
- Transformer-based sequence encoding
- Comprehensive explainability methods (local, global, thermodynamic, counterfactual)
- Dynamic optimization with real-time feedback and contradiction detection
- Advanced visualization tools for model interpretability

## Project Structure

```
PeptideAI/
├── data/                  # Data directory
│   └── Peptide.csv        # Peptide sequence and binding energy data
├── models/                # Model architecture
│   └── model.py           # Main model implementation
├── utils/                 # Utility functions
│   └── data_processing.py # Data loading and preprocessing
├── explainability/        # Explainability methods
│   └── explainers.py      # Implementation of various explainers
├── optimization/          # Dynamic optimization
│   └── dynamic_optimizer.py # Feedback loop and contradiction detection
├── visualization/         # Visualization utilities
│   └── visualizer.py      # Visualization tools
├── outputs/               # Output directory (created during runtime)
│   ├── logs/              # TensorBoard logs
│   ├── models/            # Saved models
│   ├── results/           # Evaluation results
│   └── visualizations/    # Generated visualizations
├── config.py              # Configuration settings
├── main.py                # Main script
└── requirements.txt       # Dependencies
```

## Requirements

- Python 3.9+
- PyTorch 1.9+
- CUDA-enabled GPU (recommended)
  - Tested with CUDA Version: 12.6
  - NVIDIA Driver Version: 560.35.03
- Ubuntu 20.04 (tested environment)

### Installation

We provide a setup script that will create a conda environment and install all dependencies with the correct versions for CUDA 12.6 compatibility:

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This script will:
1. Create a conda environment with Python 3.9 (if it doesn't exist)
2. Install PyTorch with CUDA 12.6 support
3. Install torchdata explicitly (required by DGL)
4. Install all other dependencies
5. Verify the installation

Alternatively, you can install dependencies manually:

```bash
# Create and activate conda environment
conda create -n py39 python=3.9
conda activate py39

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install torchdata explicitly first
pip install torchdata==0.7.0

# Install other dependencies
pip install -r requirements.txt
```

### Verifying Installation

After installation, you can verify that all dependencies are correctly installed by running the test script:

```bash
python test_installation.py
```

This script will:
1. Display version information for all dependencies
2. Verify that CUDA is available
3. Test basic DGL functionality to ensure the graph neural network components will work

If the test passes, you're ready to use the PeptideAI project.

### Troubleshooting Dependencies

#### Missing torchdata Module

If you encounter the following error:
```
ModuleNotFoundError: No module named 'torchdata.datapipes'
```

This is due to a dependency required by the project. Make sure you have torchdata installed:

```bash
pip install torchdata==0.7.0
```

#### GLIBCXX_3.4.29 Not Found Error

If you encounter the following error:
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

This is because newer versions of scipy (≥1.10.0) require a newer version of the libstdc++ library than what is available on your system. We provide several solutions:

1. **Use the fix_libstdcpp.sh script (Recommended)**:
   ```bash
   chmod +x fix_libstdcpp.sh
   ./fix_libstdcpp.sh
   ```
   This script will guide you through the process of fixing the issue.

2. **Install an older version of scipy manually**:
   ```bash
   pip uninstall -y scipy
   pip install "scipy<1.10.0"
   ```

3. **Create a new conda environment with compatible packages**:
   ```bash
   conda create -n peptideai_env -c conda-forge python=3.9 scipy=1.9.3 numpy pandas scikit-learn matplotlib seaborn tqdm networkx pyyaml
   conda activate peptideai_env
   ./setup.sh
   ```

4. **Use Docker (Advanced)**:
   Create a Dockerfile with the following content:
   ```dockerfile
   FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   WORKDIR /app
   COPY . /app
   RUN apt-get update && apt-get install -y build-essential
   RUN pip install -r requirements.txt
   RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
   ```
   Then build and run the Docker container:
   ```bash
   docker build -t peptideai .
   docker run --gpus all -it peptideai bash
   ```

#### Missing CUDA Libraries Error

If you encounter the following error:
```
OSError: libcusparse.so.12: cannot open shared object file: No such file or directory
```

This is because some CUDA libraries required by PyTorch and PyTorch Geometric are missing from your system. We provide several solutions:

1. **Use the fix_cuda_libs.sh script (Recommended)**:
   ```bash
   chmod +x fix_cuda_libs.sh
   sudo ./fix_cuda_libs.sh
   ```
   This script will install the missing CUDA libraries from the NVIDIA repository.

2. **Install the full CUDA toolkit manually**:
   ```bash
   sudo apt-get install -y cuda-toolkit-12-1
   ```
   This will install the complete CUDA toolkit, which includes all the required libraries.

3. **Use Docker (No CUDA installation required)**:
   ```bash
   docker-compose up -d
   ```
   This will run the project in a Docker container with all the required CUDA libraries pre-installed.

#### PyTorch Geometric and CUDA Integration

We now use PyTorch Geometric (PyG) instead of DGL for better CUDA integration. PyG is installed automatically by the setup script with CUDA support.

PyTorch Geometric provides excellent CUDA support and is directly built on PyTorch, offering a more seamless experience. The setup script installs PyG with CUDA 12.1 support, which is compatible with CUDA 12.6:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

For more information about the migration from DGL to PyTorch Geometric, see the `migration_plan.md` file.

### Troubleshooting Data Processing

If you encounter the following error during training:
```
ValueError: too many dimensions 'str'
```

This is due to an issue with handling string values in the data processing pipeline. We've fixed this issue in the latest version of the code by implementing one-hot encoding for categorical columns.

The fix includes:
1. Modifying the `prepare_datasets` function to perform one-hot encoding for categorical columns:
   ```python
   # Extract additional features if available
   additional_features = {}
   categorical_columns = {}
   
   for col in df.columns:
       if col not in ['sequence', 'binding_energy']:
           # Check if the column contains numeric values
           if pd.api.types.is_numeric_dtype(df[col]):
               additional_features[col] = df[col].values
           else:
               # For categorical columns (like 'solvent'), perform one-hot encoding
               print(f"Performing one-hot encoding for categorical column '{col}'")
               # Get unique values and create a mapping
               unique_values = df[col].unique()
               value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
               
               # Store the mapping for later use
               categorical_columns[col] = {
                   'mapping': value_to_idx,
                   'values': df[col].values
               }
               
               # For each unique value, create a binary feature
               for val in unique_values:
                   feature_name = f"{col}_{val}"
                   additional_features[feature_name] = (df[col] == val).astype(int).values
   ```

2. Updating the `collate_fn` function to handle the one-hot encoded features:
   ```python
   # Add additional features if available
   additional_keys = [key for key in batch[0].keys() if key not in ['sequence', 'binding_energy']]
   for key in additional_keys:
       values = [sample[key] for sample in batch]
       # Convert to tensor (all values should be numeric at this point)
       batched_sample[key] = torch.tensor(values)
   ```

This approach is more robust as it properly handles categorical data by converting it to one-hot encoded features, which can be easily processed by the model. The peptide sequences themselves are already one-hot encoded in the `create_peptide_graph` function.

We've also created a patch script (`apply_fix.py`) that can automatically apply this fix to your existing code. You can run it with:
```bash
python apply_fix.py /path/to/data_processing.py
```

## Data Format

The model expects a CSV file with the following columns:
- `sequence`: Peptide amino acid sequence
- `binding_energy`: Binding energy in kcal/mol
- Additional columns (optional): pH, temperature, solvent, experimental_method, etc.

## Usage

### Training

```bash
python main.py --mode train --data_path data/Peptide.csv --device cuda
```

### Testing

```bash
python main.py --mode test --model_path outputs/models/best_model.pth --device cuda
```

### Generating Explanations

```bash
python main.py --mode explain --model_path outputs/models/best_model.pth --explain_method integrated_gradients --device cuda
```

### Command-line Arguments

- `--mode`: Mode to run the script in (`train`, `test`, `explain`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--data_path`: Path to the data file
- `--model_path`: Path to load a pre-trained model
- `--embedding_dim`: Dimension of the embeddings
- `--num_heads`: Number of attention heads
- `--num_layers`: Number of transformer layers
- `--dropout`: Dropout rate
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of epochs
- `--early_stopping`: Patience for early stopping
- `--feedback_interval`: Interval for feedback collection
- `--contradiction_threshold`: Threshold for contradiction detection
- `--explain_method`: Explainability method (`integrated_gradients`, `deep_lift`, `gradient_shap`, `occlusion`)

## Model Architecture

The model architecture consists of several key components:

1. **Sequence Encoder**: A transformer-based encoder that processes peptide sequences.
2. **Graph Encoder**: A graph neural network (using PyTorch Geometric) that processes the structural representation of peptides.
3. **Cross-Modal Attention**: Attention mechanisms that enable information exchange between sequence and graph modalities.
4. **Multimodal Fusion**: A module that combines information from different modalities.
5. **Prediction Head**: A module that predicts binding energies based on the fused representations.

### Graph Neural Network Implementation

We use PyTorch Geometric (PyG) for implementing the graph neural network components. PyG provides several advantages:

- Native PyTorch integration for seamless tensor operations
- Excellent CUDA support for GPU acceleration
- Rich set of graph neural network layers and pooling operations
- Active development and community support

The graph encoder uses Graph Attention Networks (GAT) to process the peptide structure, capturing the relationships between amino acids and their physicochemical properties.

## Explainability Methods

PeptideAI provides several explainability methods:

1. **Local Explanation**: Explains individual predictions using methods like Integrated Gradients, DeepLIFT, GradientSHAP, and Occlusion.
2. **Residue-level Thermodynamic Maps**: Visualizes the contribution of each residue to the binding energy.
3. **Global Explanation**: Provides a global understanding of the model's behavior using feature importance analysis.
4. **Counterfactual Explanation**: Generates alternative sequences that would result in different binding energies.

## Dynamic Optimization

The model includes a dynamic optimization module with:

1. **Real-time Feedback Loop**: Collects feedback during training to improve predictions.
2. **Contradiction Detection**: Identifies and resolves contradictions in model predictions.

## Visualization

PeptideAI provides various visualization tools:

1. **Sequence Visualization**: Visualizes peptide sequences with residue contributions.
2. **Binding Energy Distribution**: Visualizes the distribution of binding energies.
3. **Peptide Graph Visualization**: Visualizes the graph representation of peptides.
4. **Attention Weights Visualization**: Visualizes attention weights in the model.
5. **Optimization Progress Visualization**: Visualizes training progress.
6. **Contradiction Detection Visualization**: Visualizes detected contradictions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was developed for research purposes to advance the field of peptide binding energy prediction and explainable AI.
