"""
Data processing utilities for the PeptideAI project.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Amino acid to index mapping for encoding sequences
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

def encode_sequence(seq):
    """
    Encode a peptide sequence to a list of indices.
    
    Args:
        seq (str): Peptide sequence.
        
    Returns:
        list: List of indices.
    """
    return [AA_TO_IDX.get(aa, 0) for aa in seq]  # Default to 0 for unknown amino acids


def load_data(data_path):
    """
    Load peptide data from CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: Loaded data.
    """
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_columns = ['sequence', 'binding_energy']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data file")
    
    # Convert sequences to uppercase
    df['sequence'] = df['sequence'].str.upper()
    
    # Filter out sequences with invalid amino acids
    valid_aa = set(AMINO_ACIDS)
    df['valid_sequence'] = df['sequence'].apply(lambda x: all(aa in valid_aa for aa in x))
    invalid_count = (~df['valid_sequence']).sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} sequences with invalid amino acids were found and will be filtered out")
        df = df[df['valid_sequence']]
    
    # Drop temporary column
    df = df.drop(columns=['valid_sequence'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} peptide sequences")
    
    return df


class PeptideDataset(Dataset):
    """
    Dataset for peptide sequences and binding energies.
    """
    def __init__(self, sequences, binding_energies, additional_features=None):
        """
        Initialize the dataset.
        
        Args:
            sequences (list): List of peptide sequences.
            binding_energies (numpy.ndarray): Array of binding energies.
            additional_features (dict, optional): Dictionary of additional features.
        """
        self.sequences = sequences
        self.binding_energies = binding_energies
        self.additional_features = additional_features or {}
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: Sample dictionary.
        """
        sample = {
            'sequence': self.sequences[idx],
            'binding_energy': self.binding_energies[idx]
        }
        
        # Add additional features if available
        for key, values in self.additional_features.items():
            sample[key] = values[idx]
        
        return sample


def prepare_datasets(df):
    """
    Prepare train, validation, and test datasets.
    
    Args:
        df (pandas.DataFrame): Data frame with peptide data.
        
    Returns:
        tuple: Train, validation, and test datasets.
    """
    # Extract sequences and binding energies
    sequences = df['sequence'].tolist()
    binding_energies = df['binding_energy'].values
    
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
    
    # Split data into train, validation, and test sets
    train_val_sequences, test_sequences, train_val_energies, test_energies = train_test_split(
        sequences, binding_energies, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    
    train_sequences, val_sequences, train_energies, val_energies = train_test_split(
        train_val_sequences, train_val_energies, 
        test_size=VAL_RATIO/(TRAIN_RATIO + VAL_RATIO), 
        random_state=RANDOM_SEED
    )
    
    # Split additional features
    train_val_indices = [sequences.index(seq) for seq in train_val_sequences]
    test_indices = [sequences.index(seq) for seq in test_sequences]
    train_indices = [sequences.index(seq) for seq in train_sequences]
    val_indices = [sequences.index(seq) for seq in val_sequences]
    
    train_additional = {}
    val_additional = {}
    test_additional = {}
    
    for key, values in additional_features.items():
        train_additional[key] = values[train_indices]
        val_additional[key] = values[val_indices]
        test_additional[key] = values[test_indices]
    
    # Create datasets
    train_dataset = PeptideDataset(train_sequences, train_energies, train_additional)
    val_dataset = PeptideDataset(val_sequences, val_energies, val_additional)
    test_dataset = PeptideDataset(test_sequences, test_energies, test_additional)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def create_peptide_graph(sequence):
    """
    Create a graph representation of a peptide sequence using PyTorch Geometric.
    
    Args:
        sequence (str): Peptide sequence.
        
    Returns:
        torch_geometric.data.Data: Graph representation of the peptide.
    """
    # Number of nodes (amino acids)
    num_nodes = len(sequence)
    
    # Create edges
    edge_index = []
    edge_attr = []
    
    # Add backbone edges (sequential connections)
    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
        edge_attr.extend([0, 0])  # Edge type 0 for backbone
    
    # Add spatial edges (connections between amino acids that are not adjacent in sequence)
    # For simplicity, we connect amino acids that are within 3 positions of each other
    for i in range(num_nodes):
        for j in range(num_nodes):
            if abs(i - j) > 1 and abs(i - j) <= 3:  # Not adjacent but within range
                edge_index.append([i, j])
                edge_attr.append(1)  # Edge type 1 for spatial
    
    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    # Node features
    # Amino acid identity
    amino_acid = torch.zeros(num_nodes, dtype=torch.long)
    for i, aa in enumerate(sequence):
        amino_acid[i] = AMINO_ACIDS.index(aa)
    
    # Physicochemical properties
    hydrophobicity = torch.zeros(num_nodes, dtype=torch.float)
    weight = torch.zeros(num_nodes, dtype=torch.float)
    charge = torch.zeros(num_nodes, dtype=torch.float)
    
    for i, aa in enumerate(sequence):
        hydrophobicity[i] = AA_PROPERTIES['hydrophobicity'].get(aa, 0)
        weight[i] = AA_PROPERTIES['weight'].get(aa, 0)
        charge[i] = AA_PROPERTIES['charge'].get(aa, 0)
    
    # Create node feature matrix (3 features: hydrophobicity, weight, charge)
    x = torch.stack([hydrophobicity, weight, charge], dim=1)
    
    # Create PyG graph
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        amino_acid=amino_acid
    )
    
    return data


def create_batch_graphs(sequences, device=None):
    """
    Create a batch of graphs for a list of peptide sequences using PyTorch Geometric.
    
    Args:
        sequences (list): List of peptide sequences.
        device (torch.device, optional): Device to move the graphs to.
        
    Returns:
        torch_geometric.data.Batch: Batched graph.
    """
    # Create individual graphs
    graphs = [create_peptide_graph(seq) for seq in sequences]
    
    # Batch graphs
    batched_graph = Batch.from_data_list(graphs)
    
    # Move to device if specified
    if device is not None:
        batched_graph = batched_graph.to(device)
    
    return batched_graph


def collate_fn(batch, device=None):
    """
    Collate function for DataLoader using PyTorch Geometric.
    
    Args:
        batch (list): List of samples.
        device (torch.device, optional): Device to move the tensors to.
        
    Returns:
        dict: Batched samples.
    """
    # Extract sequences and binding energies
    sequences = [sample['sequence'] for sample in batch]
    binding_energies = torch.tensor([sample['binding_energy'] for sample in batch], dtype=torch.float)
    
    # Create batch of graphs
    graph = create_batch_graphs(sequences, device)
    
    # Create batched sample
    batched_sample = {
        'sequences': sequences,
        'binding_energies': binding_energies,
        'graph': graph
    }
    
    # Move binding energies to device if specified
    if device is not None:
        batched_sample['binding_energies'] = batched_sample['binding_energies'].to(device)
    
    # Add additional features if available
    additional_keys = [key for key in batch[0].keys() if key not in ['sequence', 'binding_energy']]
    for key in additional_keys:
        values = [sample[key] for sample in batch]
        # Convert to tensor (all values should be numeric at this point)
        batched_sample[key] = torch.tensor(values)
        # Move to device if specified
        if device is not None:
            batched_sample[key] = batched_sample[key].to(device)
    
    return batched_sample


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE, device=None):
    """
    Get DataLoaders for train, validation, and test datasets.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size.
        device (torch.device, optional): Device to move the tensors to.
        
    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    # Create a custom collate function with the specified device
    collate_with_device = lambda b: collate_fn(b, device)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_with_device
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_with_device
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_with_device
    )
    
    return train_loader, val_loader, test_loader


def get_advanced_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE, device=None):
    """
    Get advanced DataLoaders with additional features.
    
    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size.
        device (torch.device, optional): Device to move the tensors to.
        
    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    # Create a custom collate function with the specified device
    collate_with_device = lambda b: collate_fn(b, device)
    
    # Create basic dataloaders
    if train_dataset is not None and val_dataset is not None:
        train_loader, val_loader, test_loader = get_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size, device
        )
    else:
        train_loader, val_loader = None, None
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_with_device
        )
    
    return train_loader, val_loader, test_loader


def preprocess_sequence(sequence):
    """
    Preprocess a peptide sequence for model input.
    
    Args:
        sequence (str): Peptide sequence.
        
    Returns:
        str: Preprocessed sequence.
    """
    # Convert to uppercase
    sequence = sequence.upper()
    
    # Truncate if too long
    if len(sequence) > SEQ_MAX_LENGTH:
        sequence = sequence[:SEQ_MAX_LENGTH]
    
    return sequence


def preprocess_batch(sequences, binding_energies=None, device=None):
    """
    Preprocess a batch of peptide sequences for model input using PyTorch Geometric.
    
    Args:
        sequences (list): List of peptide sequences.
        binding_energies (numpy.ndarray, optional): Array of binding energies.
        device (torch.device, optional): Device to move the tensors to.
        
    Returns:
        dict: Preprocessed batch.
    """
    # Preprocess sequences
    preprocessed_sequences = [preprocess_sequence(seq) for seq in sequences]
    
    # Create batch of graphs
    graph = create_batch_graphs(preprocessed_sequences, device)
    
    # Create batch
    batch = {
        'sequences': preprocessed_sequences,
        'graph': graph
    }
    
    # Add binding energies if provided
    if binding_energies is not None:
        batch['binding_energies'] = torch.tensor(binding_energies, dtype=torch.float)
        if device is not None:
            batch['binding_energies'] = batch['binding_energies'].to(device)
    
    return batch
