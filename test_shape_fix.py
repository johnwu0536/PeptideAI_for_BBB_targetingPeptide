"""
Test script to verify that the shape issues have been fixed.
"""

import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model import PeptideBindingModel
from utils.data_processing import create_batch_graphs, load_peptide_data
from config import *

def test_model(device='cuda'):
    """
    Test the model with different batch sizes and sequence lengths.
    """
    print("Testing model with different batch sizes and sequence lengths...")
    
    # Create model
    model = PeptideBindingModel()
    model.to(device)
    model.eval()
    
    # Test with different batch sizes
    batch_sizes = [5, 19, 32]
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        
        # Create test sequences with different lengths
        sequences = []
        for i in range(batch_size):
            # Create sequences with lengths from 5 to 15
            length = min(5 + i % 11, 15)
            seq = ''.join([AMINO_ACIDS[i % len(AMINO_ACIDS)] for i in range(length)])
            sequences.append(seq)
        
        print(f"Created {len(sequences)} sequences with lengths from {len(sequences[0])} to {len(sequences[-1])}")
        
        # Create graph
        graph = create_batch_graphs(sequences)
        graph = graph.to(device)
        
        # Forward pass
        try:
            with torch.no_grad():
                outputs = model(sequences, graph)
                binding_energy = outputs['binding_energy']
                
            print(f"Forward pass successful!")
            print(f"Binding energy shape: {binding_energy.shape}")
            print(f"Binding energy values: {binding_energy}")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll tests completed!")

def test_with_real_data(data_path, device='cuda'):
    """
    Test the model with real data.
    """
    print(f"Testing model with real data from {data_path}...")
    
    # Load data
    sequences, binding_energies, graphs = load_peptide_data(data_path)
    
    # Create model
    model = PeptideBindingModel()
    model.to(device)
    model.eval()
    
    # Test with different batch sizes
    batch_sizes = [5, 19, 32]
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        
        # Select a subset of data
        batch_sequences = sequences[:batch_size]
        batch_graphs = graphs[:batch_size].to(device)
        
        # Forward pass
        try:
            with torch.no_grad():
                outputs = model(batch_sequences, batch_graphs)
                predicted_energies = outputs['binding_energy']
                
            print(f"Forward pass successful!")
            print(f"Predicted energies shape: {predicted_energies.shape}")
            print(f"Predicted energies values: {predicted_energies}")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll tests completed!")

def main():
    parser = argparse.ArgumentParser(description='Test the model with different batch sizes and sequence lengths.')
    parser.add_argument('--data_path', type=str, default='data/Peptide.csv', help='Path to the data file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--test_type', type=str, default='both', choices=['synthetic', 'real', 'both'], help='Type of test to run.')
    
    args = parser.parse_args()
    
    # Run tests
    if args.test_type in ['synthetic', 'both']:
        test_model(device=args.device)
    
    if args.test_type in ['real', 'both']:
        test_with_real_data(args.data_path, device=args.device)

if __name__ == '__main__':
    main()
