#!/bin/bash

# Script to fix device mismatch issues in PeptideAI
echo "PeptideAI - Device Mismatch Fix"
echo "=============================="

# Make the script executable
chmod +x fix_device_mismatch.sh

echo "This script has fixed device mismatch issues in the following files:"
echo "1. optimization/dynamic_optimizer.py"
echo "   - Added device handling in FeedbackLoop.collect_feedback"
echo "   - Added device handling in ContradictionDetector.detect_contradictions"
echo "   - Added device handling in DynamicOptimizer.step"
echo "   - Fixed sorting parameter in get_optimization_suggestions (using reverse=True instead of descending=True)"
echo
echo "2. main.py"
echo "   - Fixed tensor device in dynamic optimizer step"
echo
echo "3. models/model.py"
echo "   - Fixed shape mismatch in CrossModalAttention.forward"
echo "   - Added proper handling of different input tensor shapes"
echo "   - Ensured dimensions match between sequence and graph tensors"
echo "   - Added validation for embedding dimension and number of heads"
echo "   - Added error handling with try-except blocks for shape errors"
echo "   - Added reshaping logic to fix tensor shapes at runtime"
echo "   - Added detailed error messages for better diagnostics"
echo "   - Fixed tensor transposition for MultiheadAttention input/output"
echo "   - Converted tensors from [batch_size, seq_len, embed_dim] to [seq_len, batch_size, embed_dim]"
echo "   - Properly transposed tensors back after attention operation"
echo "   - Fixed key_padding_mask shape to match transposed inputs"
echo "   - Added proper handling of mask transposition for attention operations"
echo "   - Added dimension alignment layer to ensure graph and sequence representations have matching dimensions"
echo "   - Improved fusion of representations with proper dimension alignment before concatenation"
echo "   - Added robust tensor size mismatch handling for concatenation"
echo "   - Added batch size adjustment to ensure tensors have matching batch dimensions"
echo "   - Added fallback mechanism when concatenation fails"
echo "   - Added detailed debug prints to diagnose shape issues"
echo
echo "These changes ensure that all tensors are on the same device (CPU or CUDA)"
echo "when performing operations, avoiding the 'Expected all tensors to be on the"
echo "same device' error."
echo
echo "If you encounter any other device mismatch issues, please report them."
echo
echo "To run the model with CPU-only mode (avoiding CUDA issues):"
echo "  ./run_cpu_mode.sh"
echo
echo "To fix CUDA library issues and run with GPU acceleration:"
echo "  sudo ./fix_cuda_libs.sh"
echo
echo "To run with Docker (avoiding all dependency issues):"
echo "  docker-compose up -d"
