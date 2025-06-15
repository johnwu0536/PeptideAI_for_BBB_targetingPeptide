#!/usr/bin/env python3
"""
Test script to verify that the PyTorch Geometric implementation works correctly.
"""

import os
import torch
from torch_geometric.data import Data, Batch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_processing import create_peptide_graph, create_batch_graphs
from models.model import GraphEncoder
from config import *

def test_graph_creation():
    """Test the creation of peptide graphs using PyTorch Geometric."""
    print("\n" + "="*50)
    print("TESTING GRAPH CREATION")
    print("="*50)
    
    # Create a test peptide sequence
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids
    print(f"Test sequence: {sequence}")
    
    # Create a graph
    graph = create_peptide_graph(sequence)
    print(f"Graph created: {graph}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge attributes shape: {graph.edge_attr.shape}")
    
    # Visualize the graph
    plt.figure(figsize=(10, 8))
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.title("Peptide Graph Visualization")
    
    # Save the visualization
    os.makedirs("outputs/visualizations", exist_ok=True)
    plt.savefig("outputs/visualizations/peptide_graph.png")
    print(f"Graph visualization saved to: outputs/visualizations/peptide_graph.png")
    
    # Test batch creation
    sequences = ["ACDEF", "GHIKLM", "NPQRST"]
    print(f"\nTest sequences for batch: {sequences}")
    
    # Create a batch of graphs
    batch = create_batch_graphs(sequences)
    print(f"Batch created: {batch}")
    print(f"Number of graphs in batch: {batch.num_graphs}")
    print(f"Total number of nodes: {batch.num_nodes}")
    print(f"Batch indices: {batch.batch}")
    
    print("\nGraph creation test passed!")
    print("="*50)
    
    return graph, batch

def test_graph_encoder(graph, batch):
    """Test the graph encoder using PyTorch Geometric."""
    print("\n" + "="*50)
    print("TESTING GRAPH ENCODER")
    print("="*50)
    
    # Create a graph encoder
    input_dim = 3  # hydrophobicity, weight, charge
    hidden_dim = 64
    num_layers = 2
    dropout = 0.1
    
    encoder = GraphEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    print(f"Graph encoder created: {encoder}")
    
    # Test with a single graph
    print("\nTesting with a single graph...")
    # Ensure graph is on the same device as the encoder
    device = next(encoder.parameters()).device
    graph = graph.to(device)
    
    outputs = encoder(graph)
    node_features = outputs['node_features']
    graph_repr = outputs['graph_repr']
    attention_weights = outputs['attention_weights']
    
    print(f"Node features shape: {node_features.shape}")
    print(f"Graph representation shape: {graph_repr.shape}")
    print(f"Number of attention weight matrices: {len(attention_weights)}")
    
    # Test with a batch of graphs
    print("\nTesting with a batch of graphs...")
    # Ensure batch is on the same device as the encoder
    batch = batch.to(device)
    
    batch_outputs = encoder(batch)
    batch_node_features = batch_outputs['node_features']
    batch_graph_repr = batch_outputs['graph_repr']
    batch_attention_weights = batch_outputs['attention_weights']
    
    print(f"Batch node features shape: {batch_node_features.shape}")
    print(f"Batch graph representation shape: {batch_graph_repr.shape}")
    print(f"Number of batch attention weight matrices: {len(batch_attention_weights)}")
    
    # Check that the batch graph representation has the correct shape
    assert batch_graph_repr.shape[0] == batch.num_graphs, "Batch graph representation should have one vector per graph"
    
    print("\nGraph encoder test passed!")
    print("="*50)

def test_cuda_support():
    """Test CUDA support for PyTorch Geometric."""
    print("\n" + "="*50)
    print("TESTING CUDA SUPPORT")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping CUDA test.")
        return
    
    print("CUDA is available. Testing PyTorch Geometric with CUDA...")
    
    # Create a test peptide sequence
    sequence = "ACDEFGHIKLM"
    print(f"Test sequence: {sequence}")
    
    # Create a graph
    graph = create_peptide_graph(sequence)
    
    # Move graph to CUDA
    graph_cuda = graph.to('cuda')
    print(f"Graph moved to CUDA: {graph_cuda}")
    print(f"Node features device: {graph_cuda.x.device}")
    print(f"Edge index device: {graph_cuda.edge_index.device}")
    
    # Create a graph encoder
    encoder = GraphEncoder(
        input_dim=3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ).to('cuda')
    
    # Test with a graph on CUDA
    outputs = encoder(graph_cuda)
    node_features = outputs['node_features']
    graph_repr = outputs['graph_repr']
    
    print(f"Node features device: {node_features.device}")
    print(f"Graph representation device: {graph_repr.device}")
    
    print("\nCUDA support test passed!")
    print("="*50)

def main():
    """Main function."""
    print("Testing PyTorch Geometric implementation...")
    
    # Test graph creation
    graph, batch = test_graph_creation()
    
    # Test graph encoder
    test_graph_encoder(graph, batch)
    
    # Test CUDA support
    test_cuda_support()
    
    print("\nAll tests passed! The PyTorch Geometric implementation is working correctly.")

if __name__ == "__main__":
    main()
