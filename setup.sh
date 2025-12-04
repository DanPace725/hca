#!/bin/bash
# Setup script for Hybrid Reasoning System

set -e

echo "=========================================="
echo "Hybrid Reasoning System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10 or higher required"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Ask about installation type
echo "Select installation type:"
echo "1) Minimal (rules + stub only, no dependencies)"
echo "2) Full (with PyTorch GRU and training capabilities)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "✓ Minimal installation selected"
        echo "  No dependencies needed - uses Python stdlib only"
        ;;
    2)
        echo ""
        echo "Installing dependencies..."
        
        # Detect CUDA
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU detected. Install PyTorch with CUDA support? [y/N]"
            read -p "> " cuda_choice
            case $cuda_choice in
                [Yy]*)
                    echo "Installing PyTorch with CUDA 12.1..."
                    pip install torch --index-url https://download.pytorch.org/whl/cu121
                    pip install -r requirements.txt
                    ;;
                *)
                    echo "Installing CPU-only PyTorch..."
                    pip install torch --index-url https://download.pytorch.org/whl/cpu
                    pip install -r requirements.txt
                    ;;
            esac
        else
            echo "No GPU detected. Installing CPU-only PyTorch..."
            pip install torch --index-url https://download.pytorch.org/whl/cpu
            pip install -r requirements.txt
        fi
        echo "✓ Dependencies installed"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install Ollama: https://ollama.ai"
echo "2. Pull Phi-3 model: ollama pull phi3"
echo "3. Start Ollama: ollama serve"
echo "4. Run example: python main.py 'Tom has 3 apples...' --scenario apple_sharing"
echo ""
echo "See README.md for more information."

