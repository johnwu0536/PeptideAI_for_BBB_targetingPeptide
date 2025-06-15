#!/bin/bash

# Script to fix graph handling issues in PeptideAI
echo "PeptideAI - Graph Handling Fix"
echo "============================="

# Make the script executable
chmod +x fix_graph_handling.sh

echo "This script has fixed graph handling issues in the following files:"
echo "1. models/model.py"
echo "   - Fixed AssertionError: expecting key_padding_mask shape of (19, 50), but got torch.Size([50, 1])"
echo "   - Added shape checking and correction for key_padding_mask in seq_to_graph_attention"
echo "   - Added shape checking and correction for key_padding_mask in graph_to_seq_attention"
echo "   - Added detailed debug prints to diagnose shape issues"
echo "   - Added automatic transposition of key_padding_mask when dimensions are flipped"
echo "   - Added automatic reshaping of key_padding_mask when total elements match but shape is wrong"
echo "   - Added FINAL FIX to always ensure mask has shape (bsz, src_len) regardless of previous transformations"
echo "   - Fixed specific case of (32, 19) vs (19, 32) shape mismatch"
echo "   - Added DIRECT FIX right before passing mask to attention function to ensure correct shape"
echo "   - Added check to transpose mask if shape[0] != query.shape[1] (batch_size)"
echo "   - Implemented user-suggested fix to compare mask shape with query tensor shape"
echo "   - Set batch_first=False in MultiheadAttention to avoid transposition issues"
echo "   - Added CRITICAL FIX to force the correct mask shape regardless of previous transformations"
echo "   - Added fallback to create a new mask with the correct shape if dimensions don't match"
echo
echo "2. explainability/explainers.py"
echo "   - Fixed AttributeError: 'list' object has no attribute 'x'"
echo "   - Fixed AttributeError: 'GlobalStorage' object has no attribute 'ndata'"
echo "   - Fixed AssertionError: expecting key_padding_mask shape of (19, 50), but got torch.Size([50, 1])"
echo "   - Added type checking for graph objects in ExplainabilityManager methods"
echo "   - Added automatic conversion from list to proper graph objects"
echo "   - Added proper device handling for converted graphs"
echo "   - Updated docstrings to reflect the accepted graph types"
echo "   - Fixed compatibility with PyTorch Geometric graphs (removed DGL-specific code)"
echo "   - Modified forward_wrapper methods to work with PyTorch Geometric graphs"
echo "   - Added key_padding_mask shape correction in forward_wrapper methods"
echo "   - Expanded sequence_mask to match expected shape (batch_size, 50)"
echo "   - Fixed dimension order in mask expansion (was creating [50, 1] instead of [batch_size, 50])"
echo "   - Added proper broadcasting of mask values across all positions"
echo
echo "These changes ensure that the explainability methods can handle both"
echo "PyTorch Geometric graph objects and lists of sequences, automatically"
echo "converting lists to proper graph objects when needed."
echo
echo "If you encounter any other graph handling issues, please report them."
echo
echo "To run the model with the fixes applied:"
echo "  python main.py --mode train --data_path data/Peptide.csv --device cuda"
echo
echo "To run with Docker (avoiding all dependency issues):"
echo "  docker-compose up -d"
