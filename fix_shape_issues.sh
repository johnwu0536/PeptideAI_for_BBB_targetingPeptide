#!/bin/bash

# Script to fix shape issues in PeptideAI
echo "PeptideAI - Shape Issues Fix"
echo "============================="

# Make the script executable
chmod +x fix_shape_issues.sh

echo "This script has fixed shape issues in the following files:"
echo "1. models/model.py"
echo "   - Changed MultiheadAttention modules to use batch_first=True"
echo "   - Removed unnecessary transpose operations in forward method"
echo "   - Fixed shape mismatch in seq_to_graph_attention and graph_to_seq_attention"
echo "   - Fixed 'shape [19, 136, 32] is invalid for input of size 24320' error"
echo "   - Fixed 'shape [5, 19, 256] is invalid for input of size 82688' error"
echo "   - Added proper mask handling for batch_first=True"
echo "   - Added batch size matching to prevent 'The size of tensor a (17) must match the size of tensor b (5)' error"
echo "   - Added dimension checks and adjustments throughout the forward pass"
echo
echo "2. explainability/explainers.py"
echo "   - Updated mask handling to work with batch_first=True"
echo "   - Fixed shape issues in forward_wrapper methods"
echo "   - Fixed 'Cannot choose target column with output shape torch.Size([1])' error in Captum"
echo "   - Reshaped binding_energy from 1D to 2D for Captum compatibility"
echo "   - Fixed 'GlobalStorage' object has no attribute 'device' error in CounterfactualExplainer"
echo
echo "3. visualization/visualizer.py"
echo "   - Fixed 's must be a scalar, or float array-like with the same size as x and y' error in visualize_peptide_graph"
echo "   - Added checks to ensure node_colors and node_sizes match the number of nodes in the graph"
echo "   - Added handling for unknown amino acids"
echo "   - Fixed 'GlobalStorage' object has no attribute 'nodes' error"
echo "   - Added support for PyTorch Geometric graphs in visualize_peptide_graph"
echo "   - Added fallback to create a simple chain graph if conversion fails"
echo
echo "These changes ensure that the model can handle tensors with different shapes"
echo "and properly align dimensions for attention operations."
echo
echo "If you encounter any other shape issues, please report them."
echo
echo "To run the model with the fixes applied:"
echo "  python main.py --mode train --data_path data/Peptide.csv --device cuda"
echo
echo "To run with Docker (avoiding all dependency issues):"
echo "  docker-compose up -d"
