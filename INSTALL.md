# Installation Guide

## Quick Install (For Your CUDA 11.8 GPU)

### Windows (Your Setup):
```bash
# Option 1: Automated
.\install-gpu.bat

# Option 2: Manual
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Verify Installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.x.x+cu118
CUDA available: True
```

---

## Other Configurations

### CPU Only (No GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Minimal Install (No Training, Rules Only)

No installation needed! The system works with Python 3.10+ standard library only.

Just make sure Ollama is running:
```bash
ollama pull phi3
ollama serve
```

---

## Dependencies Overview

### Must Have:
- Python 3.10+
- Ollama with phi3 model (for semantic parsing)

### For GRU Training:
- PyTorch 2.0+ (with or without CUDA)
- NumPy, Pandas
- Matplotlib, TensorBoard
- See `requirements.txt` for complete list

### For Development:
- pytest (testing)
- black, ruff (formatting/linting)
- mypy (type checking)

---

## Troubleshooting

### "CUDA not available" but you have GPU:
```bash
# Check CUDA version
nvidia-smi

# Make sure you installed the right PyTorch version
# CUDA 11.8 → cu118
# CUDA 12.1 → cu121
```

### Import errors:
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Ollama connection failed:
```bash
# Make sure Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/version
```

