"""
Test script to verify the installation of dependencies for the PeptideAI project.
"""

import os
import sys
import torch
import torchdata
import torch_geometric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import transformers
import captum
import networkx as nx

def print_dependency_info():
    """Print information about installed dependencies."""
    print("\n" + "="*50)
    print("DEPENDENCY INFORMATION")
    print("="*50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # torchdata
    print(f"\ntorchdata version: {torchdata.__version__}")
    
    # PyTorch Geometric
    print(f"\nPyTorch Geometric version: {torch_geometric.__version__}")
    # Check if PyG CUDA is available
    try:
        from torch_geometric.data import Data
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()
        x = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        if torch.cuda.is_available():
            data = data.to('cuda')
            print("PyTorch Geometric CUDA support: Available")
            print(f"Data device: {data.x.device}")
        else:
            print("PyTorch Geometric CUDA support: Not tested (CUDA not available)")
    except Exception as e:
        print(f"PyTorch Geometric CUDA support: Not available - {str(e)}")
        print("To install PyTorch Geometric with CUDA support, run:")
        print("pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html")
    
    # Other dependencies
    print(f"\nNumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Captum version: {captum.__version__}")
    print(f"NetworkX version: {nx.__version__}")
    
    print("\n" + "="*50)

def test_pyg_functionality():
    """Test basic PyTorch Geometric functionality."""
    print("\n" + "="*50)
    print("TESTING PYTORCH GEOMETRIC FUNCTIONALITY")
    print("="*50)
    
    # Import PyG modules
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    
    # Create a simple graph
    print("Creating a simple graph...")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(3, 5)  # 3 nodes with 5 features each
    data = Data(x=x, edge_index=edge_index)
    print(f"Graph created: {data}")
    
    # Test CUDA support if available
    if torch.cuda.is_available():
        try:
            print("\nTesting PyTorch Geometric CUDA support...")
            data_cuda = data.to('cuda')
            print(f"Graph moved to CUDA: {data_cuda.x.device}")
        except Exception as e:
            print(f"Failed to move graph to CUDA: {str(e)}")
            print("To install PyTorch Geometric with CUDA support, run:")
            print("pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html")
    
    # Add edge features
    print("\nAdding edge features...")
    data.edge_attr = torch.randn(data.edge_index.size(1), 2)
    print(f"Edge features added: {data.edge_attr}")
    
    # Test message passing with a GCN layer
    print("\nTesting message passing with GCN...")
    gcn = GCNConv(5, 10)
    
    # Move everything to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = gcn.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    out = gcn(data.x, data.edge_index)
    print(f"GCN output shape: {out.shape}")
    print(f"GCN output device: {out.device}")
    
    # Test global pooling
    print("\nTesting global pooling...")
    # Create a batch with two graphs
    batch = torch.tensor([0, 0, 0, 1, 1], device=device)  # First 3 nodes belong to graph 0, last 2 to graph 1
    x_batch = torch.randn(5, 10, device=device)  # 5 nodes with 10 features each
    pooled = global_mean_pool(x_batch, batch)
    print(f"Pooled output shape: {pooled.shape}")
    print(f"Pooled output device: {pooled.device}")
    
    print("\nPyTorch Geometric functionality test passed!")
    print("="*50)

def main():
    """Main function."""
    print("\nTesting PeptideAI dependencies...")
    
    # Print dependency information
    print_dependency_info()
    
    # Test PyTorch Geometric functionality
    test_pyg_functionality()
    
    print("\nAll tests passed! The environment is correctly set up for PeptideAI.")

if __name__ == "__main__":
    main()
