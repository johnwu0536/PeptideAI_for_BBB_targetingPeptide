#!/usr/bin/env python3
"""
Patch script to fix the 'too many dimensions 'str'' error in data_processing.py.
This script will modify the data_processing.py file to properly handle string values.
"""

import os
import re
import sys

def apply_patch(file_path):
    """Apply the patch to the data_processing.py file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the prepare_datasets function to add one-hot encoding for categorical columns
    prepare_datasets_pattern = r"    # Extract additional features if available\n    additional_features = \{\}\n    for col in df\.columns:\n        if col not in \['sequence', 'binding_energy'\]:\n            additional_features\[col\] = df\[col\]\.values"
    if prepare_datasets_pattern in content:
        fixed_prepare_datasets_code = """    # Extract additional features if available
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
                    additional_features[feature_name] = (df[col] == val).astype(int).values"""
        content = content.replace(prepare_datasets_pattern, fixed_prepare_datasets_code)
    else:
        print("Warning: Could not find prepare_datasets pattern to fix")
    
    # Update the collate_fn function to handle the one-hot encoded features
    collate_pattern = r"    # Add additional features if available\n    additional_keys = \[key for key in batch\[0\]\.keys\(\) if key not in \['sequence', 'binding_energy'\]\]\n    for key in additional_keys:\n        batched_sample\[key\] = torch\.tensor\(\[sample\[key\] for sample in batch\]\)"
    if collate_pattern in content:
        fixed_collate_code = """    # Add additional features if available
    additional_keys = [key for key in batch[0].keys() if key not in ['sequence', 'binding_energy']]
    for key in additional_keys:
        values = [sample[key] for sample in batch]
        # Convert to tensor (all values should be numeric at this point)
        batched_sample[key] = torch.tensor(values)"""
        content = content.replace(collate_pattern, fixed_collate_code)
    else:
        print("Warning: Could not find collate_fn pattern to fix")
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched: {file_path}")
    return True

def main():
    """Main function."""
    # Try to find data_processing.py in common locations
    possible_paths = [
        "utils/data_processing.py",
        "../utils/data_processing.py",
        "PeptideAI/utils/data_processing.py",
        "../PeptideAI/utils/data_processing.py",
        "PeptideAI.3/utils/data_processing.py",
        "../PeptideAI.3/utils/data_processing.py",
    ]
    
    # Check if a path was provided as an argument
    if len(sys.argv) > 1:
        possible_paths.insert(0, sys.argv[1])
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found data_processing.py at: {path}")
            if apply_patch(path):
                print("\nPatch applied successfully!")
                print("The 'too many dimensions 'str'' error should now be fixed.")
                
                # Add instructions for installing PyTorch Geometric with CUDA support
                print("\nIMPORTANT: We now use PyTorch Geometric instead of DGL for better CUDA integration.")
                print("If you encounter CUDA compatibility issues with graph neural networks, install PyG with CUDA support:")
                print("\nRun the following command:")
                print("pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html")
                print("\nFor more information about the migration from DGL to PyTorch Geometric, see the migration_plan.md file.")
                
                print("\nYou can now run your code again.")
                return 0
    
    print("\nError: Could not find data_processing.py in any of the expected locations.")
    print("Please provide the path to data_processing.py as an argument:")
    print("python apply_fix.py /path/to/data_processing.py")
    return 1

if __name__ == "__main__":
    sys.exit(main())
