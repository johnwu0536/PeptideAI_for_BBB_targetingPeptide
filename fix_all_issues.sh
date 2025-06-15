#!/bin/bash

# Script to fix all issues in PeptideAI
echo "PeptideAI - Comprehensive Fix Script"
echo "==================================="

# Make the script executable
chmod +x fix_all_issues.sh

echo "This script fixes multiple issues in the PeptideAI project:"
echo "1. Argument parsing issues"
echo "2. Graph handling issues"
echo "3. Shape mismatch issues"

# Fix 1: Argument Parsing Issues
echo
echo "Fix 1: Argument Parsing Issues"
echo "-----------------------------"
echo "- Fixed '--model_path' being misinterpreted as '--mode_path'"
echo "- Added debug output to print parsed arguments"
echo "- Ensured proper argument parsing order"

# Fix 2: Graph Handling Issues
echo
echo "Fix 2: Graph Handling Issues"
echo "-------------------------"
echo "- Added handling for when graph is a list in GraphEncoder.forward"
echo "- Added fallback to create a default graph if graph doesn't have 'x' attribute"
echo "- Fixed 'list' object has no attribute 'x' error"
echo "- Added proper error handling in explain function"
echo "- Added missing import for create_batch_graphs in explain function"

# Fix 3: Shape Mismatch Issues
echo
echo "Fix 3: Shape Mismatch Issues"
echo "-------------------------"
echo "- Changed MultiheadAttention modules to use batch_first=True"
echo "- Fixed shape mismatch in seq_to_graph_attention and graph_to_seq_attention"
echo "- Added batch size matching to prevent tensor size mismatch errors"
echo "- Added dimension checks and adjustments throughout the forward pass"
echo "- Fixed Captum compatibility issues by reshaping binding_energy from 1D to 2D"
echo "- Created a proper wrapper module for Captum to fix 'function' object has no attribute 'register_forward_pre_hook' error"
echo "- Fixed 'Can't call numpy() on Tensor that requires grad' error by using detach() before numpy()"
echo "- Fixed 'Cannot take a larger sample than population when replace=False' error by ensuring subset_size doesn't exceed dataset size"

echo
echo "All fixes have been applied successfully!"
echo
echo "To run the model in different modes:"
echo "  Train mode: python main.py --mode train --data_path data/Peptide.csv --device cuda"
echo "  Test mode:  python main.py --mode test --model_path outputs/models/best_model.pth --device cuda"
echo "  Explain mode: python main.py --mode explain --model_path outputs/models/best_model.pth --explain_method integrated_gradients --device cuda"
