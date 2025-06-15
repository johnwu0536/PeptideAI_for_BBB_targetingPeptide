#!/usr/bin/env python3
"""
Test script that writes its output to a file.
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from utils.data_processing import prepare_datasets, collate_fn

def main():
    """Main function."""
    # Redirect stdout to a file
    with open('test_output.txt', 'w') as f:
        # Save the original stdout
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print("Testing data processing code...")
            
            # Create test data
            print("Creating test data...")
            data = {
                'sequence': ['ACDEFG', 'GHIKLM', 'NOPQRS'],
                'binding_energy': [1.0, 2.0, 3.0],
                'numeric_feature': [4.0, 5.0, 6.0],
                'string_feature': ['a', 'b', 'c']
            }
            df = pd.DataFrame(data)
            print(f"DataFrame created with shape: {df.shape}")
            
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
            
            # Check that non-numeric columns were filtered out
            if 'string_feature' not in train_dataset.additional_features:
                print("\nSuccess: Non-numeric column 'string_feature' was filtered out")
            else:
                print("\nError: Non-numeric column 'string_feature' was not filtered out")
            
            # Test collate_fn
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
            except Exception as e:
                print(f"\nError in collate_fn: {e}")
            
            print("\nAll tests completed!")
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()
