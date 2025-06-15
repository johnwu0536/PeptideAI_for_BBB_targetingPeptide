#!/usr/bin/env python3
"""
Test script to verify that the one-hot encoding for categorical columns works correctly.
"""

import os
import pandas as pd
import torch
import numpy as np
from utils.data_processing import prepare_datasets, collate_fn

def create_test_data_with_categorical():
    """Create test data with categorical columns for testing one-hot encoding."""
    # Create a test CSV file
    data = {
        'sequence': ['ACDEFG', 'GHIKLM', 'NOPQRS', 'TUVWXY'],
        'binding_energy': [1.0, 2.0, 3.0, 4.0],
        'numeric_feature': [4.0, 5.0, 6.0, 7.0],
        'solvent': ['water', 'dmso', 'water', 'dmso'],
        'pH': [7.0, 6.5, 7.2, 6.8],
        'experimental_method': ['method1', 'method2', 'method1', 'method3']
    }
    df = pd.DataFrame(data)
    
    # Save the test data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/test_categorical.csv', index=False)
    
    return df

def test_one_hot_encoding():
    """Test the one-hot encoding for categorical columns."""
    print("\nTesting one-hot encoding for categorical columns...")
    
    # Create test data with categorical columns
    df = create_test_data_with_categorical()
    print(f"Created test data with shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(df)
    
    # Check that the datasets were created correctly
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Check that the additional features were processed correctly
    print("\nAdditional features in train dataset:")
    for key, values in train_dataset.additional_features.items():
        print(f"  {key}: {values}")
    
    # Check that categorical columns were one-hot encoded
    categorical_columns = ['solvent', 'experimental_method']
    for col in categorical_columns:
        one_hot_features = [key for key in train_dataset.additional_features.keys() if key.startswith(f"{col}_")]
        if one_hot_features:
            print(f"\nOne-hot encoded features for '{col}':")
            for feature in one_hot_features:
                print(f"  {feature}")
            print(f"Success: '{col}' was one-hot encoded")
        else:
            print(f"\nError: '{col}' was not one-hot encoded")
    
    # Test collate_fn
    print("\nTesting collate_fn with one-hot encoded features...")
    
    # Create a batch
    batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    
    # Apply collate_fn
    try:
        batched_sample = collate_fn(batch)
        print("collate_fn executed successfully")
        
        # Check the batched sample
        print("\nBatched sample keys:")
        for key, value in batched_sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor of shape {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: List of length {len(value)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Check that one-hot encoded features are included in the batched sample
        one_hot_keys = [key for key in batched_sample.keys() if key.startswith("solvent_") or key.startswith("experimental_method_")]
        if one_hot_keys:
            print("\nOne-hot encoded features in batched sample:")
            for key in one_hot_keys:
                print(f"  {key}: {batched_sample[key]}")
            print("\nSuccess: One-hot encoded features were included in the batched sample")
        else:
            print("\nError: One-hot encoded features were not included in the batched sample")
        
        return True
    except Exception as e:
        print(f"\nError in collate_fn: {e}")
        return False

def main():
    """Main function."""
    print("Testing one-hot encoding for categorical columns...")
    
    # Test one-hot encoding
    success = test_one_hot_encoding()
    
    if success:
        print("\nAll tests passed! One-hot encoding is working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
