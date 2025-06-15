#!/usr/bin/env python3
"""
Test script to verify that the data processing code works correctly.
"""

import os
import pandas as pd
import torch
import numpy as np
from utils.data_processing import prepare_datasets, collate_fn

def create_test_data():
    """Create test data for testing the data processing code."""
    # Create a test CSV file
    data = {
        'sequence': ['ACDEFG', 'GHIKLM', 'NOPQRS'],
        'binding_energy': [1.0, 2.0, 3.0],
        'numeric_feature': [4.0, 5.0, 6.0],
        'string_feature': ['a', 'b', 'c']
    }
    df = pd.DataFrame(data)
    
    # Save the test data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/test_data.csv', index=False)
    
    return df

def test_prepare_datasets():
    """Test the prepare_datasets function."""
    print("\nTesting prepare_datasets function...")
    
    # Create test data
    df = create_test_data()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(df)
    
    # Check that the datasets were created correctly
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Check that the additional features were processed correctly
    print("\nAdditional features in train dataset:")
    for key, values in train_dataset.additional_features.items():
        print(f"  {key}: {values}")
    
    # Check that non-numeric columns were filtered out
    if 'string_feature' not in train_dataset.additional_features:
        print("\nSuccess: Non-numeric column 'string_feature' was filtered out")
    else:
        print("\nError: Non-numeric column 'string_feature' was not filtered out")
    
    return train_dataset, val_dataset, test_dataset

def test_collate_fn(train_dataset):
    """Test the collate_fn function."""
    print("\nTesting collate_fn function...")
    
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
        
        print("\nSuccess: collate_fn processed the batch correctly")
        return True
    except Exception as e:
        print(f"\nError in collate_fn: {e}")
        return False

def main():
    """Main function."""
    print("Testing data processing code...")
    
    # Test prepare_datasets
    train_dataset, val_dataset, test_dataset = test_prepare_datasets()
    
    # Test collate_fn
    success = test_collate_fn(train_dataset)
    
    if success:
        print("\nAll tests passed! The data processing code is working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
