# Migration Plan: Replacing DGL with PyTorch Geometric (PyG)

## Overview

This document outlines the plan to replace the Deep Graph Library (DGL) with PyTorch Geometric (PyG) in the PeptideAI project. PyG is a well-established library for graph neural networks that integrates seamlessly with PyTorch and has excellent CUDA support.

## Advantages of PyTorch Geometric

1. **Native PyTorch Integration**: PyG is built directly on PyTorch, providing a more seamless experience
2. **Excellent CUDA Support**: PyG has robust CUDA support and is actively maintained
3. **Rich Graph Neural Network Implementations**: PyG offers a wide range of GNN layers and models
4. **Active Community**: PyG has a large and active community, with frequent updates and improvements
5. **Extensive Documentation**: PyG has comprehensive documentation and tutorials

## Migration Steps

### 1. Update Dependencies

- Remove DGL from requirements.txt
- Add PyTorch Geometric and its dependencies to requirements.txt
- Update setup.sh to install PyG with CUDA support

### 2. Update Data Processing

In `utils/data_processing.py`:

- Replace `create_peptide_graph` to create PyG graphs instead of DGL graphs
- Replace `create_batch_graphs` to use PyG's batching mechanism
- Update `collate_fn` to work with PyG graphs

### 3. Update Model Architecture

In `models/model.py`:

- Replace `GraphEncoder` class to use PyG's GNN layers (e.g., `GATConv` from PyG)
- Update `PeptideBindingModel` to work with PyG graphs
- Replace DGL-specific operations with PyG equivalents

### 4. Update Testing

- Update `test_installation.py` to check for PyG instead of DGL
- Create tests to verify PyG functionality

## Implementation Details

### PyG Installation

PyG can be installed with CUDA support using:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```

Where `${TORCH_VERSION}` is the PyTorch version (e.g., 2.1.0) and `${CUDA_VERSION}` is the CUDA version (e.g., cu121).

### Graph Creation in PyG

In PyG, graphs are represented using the `Data` class:

```python
from torch_geometric.data import Data

def create_peptide_graph(sequence):
    # Number of nodes (amino acids)
    num_nodes = len(sequence)
    
    # Create edges
    edge_index = []
    edge_attr = []
    
    # Add backbone edges
    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
        edge_attr.extend([0, 0])  # Edge type 0 for backbone
    
    # Add spatial edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if abs(i - j) > 1 and abs(i - j) <= 3:
                edge_index.append([i, j])
                edge_attr.append(1)  # Edge type 1 for spatial
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    # Node features
    x = torch.zeros((num_nodes, 3), dtype=torch.float)  # 3 features: hydrophobicity, weight, charge
    amino_acid = torch.zeros(num_nodes, dtype=torch.long)
    
    for i, aa in enumerate(sequence):
        amino_acid[i] = AMINO_ACIDS.index(aa)
        x[i, 0] = AA_PROPERTIES['hydrophobicity'].get(aa, 0)
        x[i, 1] = AA_PROPERTIES['weight'].get(aa, 0)
        x[i, 2] = AA_PROPERTIES['charge'].get(aa, 0)
    
    # Create PyG graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        amino_acid=amino_acid
    )
    
    return graph
```

### Graph Neural Networks in PyG

In PyG, GNN layers are implemented differently:

```python
from torch_geometric.nn import GATConv as PyGGATConv

class GraphEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=GNN_HIDDEN_DIM, num_layers=GNN_NUM_LAYERS, dropout=GNN_DROPOUT):
        super(GraphEncoder, self).__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolutional layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(PyGGATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                dropout=dropout,
                concat=False
            ))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        # Get node features
        x = data.x
        
        # Apply node embedding
        x = self.node_embedding(x)
        
        # Apply graph convolutional layers
        attention_weights = []
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Apply GNN layer
            x, attention = gnn_layer(x, data.edge_index, return_attention_weights=True)
            
            # Apply layer normalization
            x = layer_norm(x)
            
            # Apply dropout
            x = self.dropout(x)
            
            # Store attention weights
            attention_weights.append(attention)
        
        # Get graph representation (average of node features)
        graph_repr = global_mean_pool(x, data.batch)
        
        return {
            'node_features': x,
            'graph_repr': graph_repr,
            'attention_weights': attention_weights
        }
```

## Timeline

1. **Setup and Dependencies** (Day 1)
   - Update requirements.txt and setup.sh
   - Create test environment to verify PyG installation

2. **Data Processing Migration** (Day 2)
   - Update graph creation functions
   - Update data loading and batching

3. **Model Architecture Migration** (Day 3-4)
   - Update GraphEncoder
   - Update PeptideBindingModel
   - Test model with sample data

4. **Testing and Validation** (Day 5)
   - Comprehensive testing
   - Performance comparison with DGL

5. **Documentation and Finalization** (Day 6)
   - Update README.md
   - Create migration guide for users

## Conclusion

Migrating from DGL to PyTorch Geometric will provide better CUDA integration and potentially improve performance. The migration process requires careful updates to the graph creation, data processing, and model architecture components, but the end result will be a more robust and maintainable codebase.
