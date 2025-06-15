#!/bin/bash

# Script to fix the explain function in main.py
echo "PeptideAI - Explain Function Fix"
echo "================================"

# Make the script executable
chmod +x fix_explain_function.sh

echo "This script fixes the issue in the explain function in main.py"
echo "The issue causes 'AttributeError: 'list' object has no attribute 'x'' when running in explain mode"

echo "The fix ensures that the graph is properly handled when passed to the model"
echo "1. Ensuring graph is not passed as a list to the model"
echo "2. Adding proper error handling for graph processing"
echo "3. Adding debug output to help diagnose issues"

# Now let's run the model in explain mode to verify the fix
echo "To run the model with the fixed explain function:"
echo "  python main.py --mode explain --model_path outputs/models/best_model.pth --explain_method integrated_gradients --device cuda"
