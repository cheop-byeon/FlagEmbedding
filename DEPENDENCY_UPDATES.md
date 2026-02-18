# Dependency Management Update Summary

## Overview

The FlagEmbedding repository has been updated with comprehensive dependency management following modern Python packaging standards.

## Files Modified/Created

### 1. **setup.py** (Updated)
- Updated description to be more descriptive
- Added `python_requires='>=3.8'`
- Added all missing core dependencies found through code analysis:
  - `numpy`
  - `pandas`
  - `tqdm`
  - `huggingface_hub`
  - `regex`
  - `packaging`
- Organized `extras_require` into logical groups:
  - `finetune`: For fine-tuning (deepspeed, flash-attn)
  - `evaluation-core`: Core evaluation tools (pytrec_eval, faiss-cpu)
  - `evaluation-all`: All benchmark evaluation tools
  - Individual benchmark options: `mteb`, `beir`, `air-bench`
  - `dev`: Development tools

### 2. **pyproject.toml** (New)
- Modern Python packaging standard (PEP 517/518)
- Build system configuration with setuptools
- Comprehensive project metadata
- All dependencies organized clearly
- Tool configuration for black formatter
- Python version classifiers for PyPI

### 3. **requirements.txt** (New)
- Core dependencies for basic usage
- Can be used for simple `pip install -r requirements.txt`

### 4. **requirements-finetune.txt** (New)
- Extends core requirements
- Includes: deepspeed, flash-attn

### 5. **requirements-eval-core.txt** (New)
- Core evaluation dependencies
- Includes: pytrec_eval, faiss-cpu
- Essential for BEIR, MSMARCO, MIRACL, MLDR, MKQA evaluations

### 6. **requirements-eval-all.txt** (New)
- All evaluation benchmark dependencies
- Includes MTEB, BEIR, AIR-Bench support

### 7. **requirements-dev.txt** (New)
- Complete development environment
- Includes all optional dependencies plus dev tools

### 8. **INSTALLATION.md** (New)
- Comprehensive installation guide
- Scenarios for different use cases
- Troubleshooting section
- Hardware-specific installation options

## Dependencies Discovered and Added

### Core Dependencies (were missing)
- `numpy`: Data manipulation
- `pandas`: Dataframe operations in evaluation
- `tqdm`: Progress bars
- `huggingface_hub`: Model downloading
- `regex`: Text processing in MKQA evaluation
- `packaging`: Version comparison utilities

### Evaluation Dependencies (conditional)
- `pytrec_eval`: Metric computation for information retrieval
- `faiss-cpu/gpu`: Vector similarity search
- `mteb>=1.15.0`: Massive Text Embedding Benchmark
- `beir`: BEIR benchmark suite
- `air-benchmark`: AIR-Bench evaluation

### Fine-tuning Dependencies (conditional)
- `deepspeed`: Distributed training
- `flash-attn`: Efficient attention implementation

## Installation Methods

Users can now install FlagEmbedding in multiple ways:

### Basic (Inference only)
```bash
pip install FlagEmbedding
pip install -r requirements.txt
```

### With Fine-tuning
```bash
pip install FlagEmbedding[finetune]
pip install -r requirements-finetune.txt
```

### With Evaluation
```bash
pip install FlagEmbedding[evaluation-core]    # Minimal
pip install FlagEmbedding[evaluation-all]     # Full
pip install -r requirements-eval-core.txt     # Minimal
pip install -r requirements-eval-all.txt      # Full
```

### Full Development
```bash
pip install FlagEmbedding[dev,finetune,evaluation-all]
pip install -r requirements-dev.txt
```

### Individual Benchmarks
```bash
pip install FlagEmbedding[mteb]
pip install FlagEmbedding[beir]
pip install FlagEmbedding[air-bench]
```

## Benefits of This Update

1. **Clear Dependencies**: All dependencies are now explicitly listed with clear grouping
2. **Modern Standards**: Uses `pyproject.toml` following PEP 517/518
3. **Flexible Installation**: Users can install only what they need
4. **Better Maintenance**: Easier to track and update dependencies
5. **Clear Documentation**: Installation guide for all scenarios
6. **Hardware Support**: Guides for CPU/GPU FAISS installation
7. **Troubleshooting**: Help for common installation issues

## Backward Compatibility

The update maintains backward compatibility:
- Existing installations continue to work
- `setup.py` still functions as before
- Python 3.8+ is supported

## Next Steps

1. Users should run `pip install -r requirements.txt` to install core dependencies
2. For evaluation, install `pip install -r requirements-eval-core.txt` or the specific benchmark
3. For development, use `pip install -r requirements-dev.txt`
4. Refer to `INSTALLATION.md` for detailed guidance on specific scenarios
