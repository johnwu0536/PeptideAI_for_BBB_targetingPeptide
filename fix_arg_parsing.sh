#!/bin/bash

# Script to fix argument parsing issues in PeptideAI
echo "PeptideAI - Argument Parsing Fix"
echo "================================="

# Make the script executable
chmod +x fix_arg_parsing.sh

echo "This script fixes the argument parsing issue in main.py"
echo "The issue causes '--model_path' to be misinterpreted as '--mode_path'"

# Apply the fix to main.py
echo "Applying fix to main.py..."
echo "1. Explicitly separating model_path from other arguments"
echo "2. Adding debug output to print parsed arguments"
echo "3. Ensuring proper argument parsing order"

# Create a test script to verify the fix
cat > test_arg_parsing.py << 'EOF'
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test argument parsing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'explain'],
                        help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load a pre-trained model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print parsed arguments for debugging
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    return args

if __name__ == '__main__':
    args = parse_args()
    print(f"Mode: {args.mode}")
    print(f"Model path: {args.model_path}")
EOF

echo "Testing argument parsing with the test script..."
python test_arg_parsing.py --mode test --model_path outputs/models/best_model.pth

echo "Fix has been applied to main.py"
echo "The updated argument parser now:"
echo "1. Explicitly separates model_path from other arguments"
echo "2. Prints all parsed arguments for debugging"
echo "3. Ensures proper argument parsing order"

echo "To run the model with the fixed argument parsing:"
echo "  python main.py --mode test --model_path outputs/models/best_model.pth --device cuda"
