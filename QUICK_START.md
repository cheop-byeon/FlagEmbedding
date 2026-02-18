# Quick Installation Reference

This repository is for **fine-tuning dense retrieval and reranking models**.

## One-liners by Use Case

| Use Case | Command |
|----------|---------|
| **Fine-tuning (main focus)** | `pip install FlagEmbedding[finetune]` |
| **Inference only** | `pip install FlagEmbedding` |
| **Fine-tuning + Evaluation** | `pip install FlagEmbedding[finetune,eval]` |
| **Development** | `pip install -e .[finetune,eval,dev]` |

## Requirements Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Core dependencies only |
| `requirements-finetune.txt` | Fine-tuning support (recommended) |
| `requirements-eval-core.txt` | Add evaluation tools (optional) |
| `requirements-dev.txt` | Full development environment |

## Installation Cheat Sheet

```bash
# Install core only
pip install -r requirements.txt

# Add fine-tuning support (recommended)
pip install -r requirements-finetune.txt

# Add evaluation tools (optional)
pip install -r requirements-eval-core.txt

# Install in development mode
pip install -e .[finetune]
pip install -e .[finetune,eval,dev]  # Full setup
```

## Fine-tuning Quick Start

```bash
# 1. Install with fine-tuning support
pip install FlagEmbedding[finetune]

# 2. Download datasets
python download_CodeConvo.py --split train
python download_RFCAlign.py

# 3. Run fine-tuning script
bash examples/finetune/embedder/decoder_only/ft_CodeConvo_decoder.sh
```

## Common Issues & Solutions

### PyTorch CUDA version mismatch
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1  
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### DeepSpeed installation fails
```bash
pip install deepspeed --no-build-isolation
```

### Flash Attention not working
```bash
# Ensure CUDA version is compatible
pip install flash-attn --no-build-isolation
```

## Verify Installation

```python
from FlagEmbedding import FlagModel
print("✓ Core FlagEmbedding installed")

import deepspeed
print("✓ DeepSpeed installed")

import flash_attn
print("✓ Flash Attention installed")
```

## Documentation Files

- `INSTALLATION.md` - Detailed installation guide
- `README.md` - Project overview and datasets
- `examples/finetune/` - Fine-tuning example scripts
- `FlagEmbedding/DATA_PATH_USAGE.md` - Data configuration

## Next Steps

1. Choose an example fine-tuning script from [examples/finetune](examples/finetune/)
2. Download datasets (see [README.md](README.md))
3. Configure fine-tuning parameters
4. Run the training script
