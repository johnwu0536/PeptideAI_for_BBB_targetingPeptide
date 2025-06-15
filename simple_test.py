#!/usr/bin/env python3
"""
Simple test script.
"""

print("This is a simple test script.")
print("If you can see this message, the script is working.")

# Try to import the modules
try:
    import pandas as pd
    import torch
    import numpy as np
    from utils.data_processing import prepare_datasets, collate_fn
    print("All imports successful!")
except Exception as e:
    print(f"Error importing modules: {e}")

# Create a simple test
try:
    # Create a simple DataFrame
    df = pd.DataFrame({
        'sequence': ['ACDEFG', 'GHIKLM'],
        'binding_energy': [1.0, 2.0],
        'numeric_feature': [4.0, 5.0]
    })
    print("DataFrame created successfully!")
    
    # Try to prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(df)
    print("Datasets prepared successfully!")
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Try to create a batch
    batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    print("Batch created successfully!")
    
    # Try to use collate_fn
    batched_sample = collate_fn(batch)
    print("collate_fn executed successfully!")
    print("All tests passed!")
except Exception as e:
    print(f"Error during test: {e}")
