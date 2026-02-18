# Installation Guide

This document provides instructions for installing FlagEmbedding for fine-tuning IR and reranker models.

## Primary Use Case: Fine-tuning

This repository is designed for **fine-tuning dense retrieval and reranking models**. The main dependencies support the fine-tuning workflow.

### Basic Installation (Recommended)

For fine-tuning with recommended tools:

```bash
pip install FlagEmbedding[finetune]
# or
pip install -r requirements-finetune.txt
```

This includes:
- `deepspeed`: For efficient distributed training
- `flash-attn`: For optimized attention computation

### Core Installation

If you only need to run inference without fine-tuning:

```bash
pip install FlagEmbedding
# or
pip install -r requirements.txt
```

### From Source

For development or custom modifications:

```bash
git clone <your-repo-url>
cd FlagEmbedding
pip install -e .[finetune]
```

## Optional Dependencies

### Evaluation Support

To evaluate fine-tuned models (optional):

```bash
pip install FlagEmbedding[eval]
# or
pip install -r requirements-eval-core.txt
```

This includes:
- `pytrec_eval`: For information retrieval metrics
- `faiss-cpu`: For similarity search

**Note**: For GPU-accelerated search, use `faiss-gpu` instead:
```bash
pip install faiss-gpu
```

### Development Dependencies

For development and testing:

```bash
pip install FlagEmbedding[dev]
# or
pip install -r requirements-dev.txt
```

This includes:
- `pytest`: For unit testing
- `black`: For code formatting
- `flake8`: For linting
- `isort`: For import sorting

## Troubleshooting

### PyTorch Installation

Ensure you have the correct PyTorch version for your system:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issues with `pytrec_eval` installation

If you encounter issues installing `pytrec_eval`, try the alternative:

```bash
pip install pytrec-eval-terrier
```

### FAISS Installation

For different hardware setups:

```bash
# CPU only (default)
pip install faiss-cpu

# GPU support (CUDA)
pip install faiss-gpu

# From wheel for specific CUDA version
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### DeepSpeed Issues

If you have issues with DeepSpeed installation, ensure you have CUDA development tools:

```bash
pip install deepspeed --no-build-isolation
```

## Quick Installation Reference

| Scenario | Command |
|----------|---------|
| **Fine-tuning (recommended)** | `pip install FlagEmbedding[finetune]` |
| **Inference only** | `pip install FlagEmbedding` |
| **Fine-tuning + Evaluation** | `pip install FlagEmbedding[finetune,eval]` |
| **Full setup** | `pip install FlagEmbedding[finetune,eval,dev]` |
| **From requirements file** | `pip install -r requirements-finetune.txt` |

## Common Installation Scenarios

### Scenario 1: Just run inference on pre-trained models

```bash
pip install FlagEmbedding
```

### Scenario 2: Fine-tune models (main use case)

```bash
pip install FlagEmbedding[finetune]
```

### Scenario 3: Fine-tune and evaluate

```bash
pip install FlagEmbedding[finetune,eval]
```

### Scenario 4: Development setup

```bash
git clone <your-repo-url>
cd FlagEmbedding
pip install -e .[finetune,eval,dev]
```

## Verifying Installation

To verify that FlagEmbedding is properly installed:

```python
from FlagEmbedding import FlagModel
print("✓ FlagEmbedding is properly installed")
```

For fine-tuning support:

```python
try:
    import deepspeed
    print("✓ DeepSpeed is installed")
except ImportError:
    print("✗ DeepSpeed is not installed")

try:
    import flash_attn
    print("✓ Flash Attention is installed")
except ImportError:
    print("✗ Flash Attention is not installed")
```

## Python Version Requirements

FlagEmbedding supports Python 3.8 and above. We recommend using Python 3.10 or 3.11 for best compatibility with fine-tuning tools.

```bash
python --version
```

## Next Steps

- See [QUICK_START.md](QUICK_START.md) for fine-tuning quick start
- Check [examples/finetune](examples/finetune) for example scripts
- Refer to [README.md](README.md) for dataset information
