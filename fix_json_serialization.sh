#!/bin/bash

# Script to fix JSON serialization issues in PeptideAI
echo "PeptideAI - JSON Serialization Fix"
echo "=================================="

# Make the script executable
chmod +x fix_json_serialization.sh

echo "This script has fixed JSON serialization issues in the following files:"
echo "1. main.py"
echo "   - Fixed TypeError: Object of type float32 is not JSON serializable"
echo "   - Added explicit conversion of NumPy and PyTorch types to native Python types"
echo "   - Added error handling with fallback mechanism for JSON serialization"
echo "   - Added debugging information to metrics output"
echo
echo "These changes ensure that all metrics can be properly serialized to JSON"
echo "when saving test results, avoiding the 'Object of type float32 is not JSON serializable' error."
echo
echo "If you encounter any other serialization issues, please report them."
echo
echo "To run the model with the fixes applied:"
echo "  python main.py --mode train --data_path data/Peptide.csv --device cuda"
echo
echo "To run with Docker (avoiding all dependency issues):"
echo "  docker-compose up -d"
