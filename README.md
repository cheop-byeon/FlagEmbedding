# Finetuned Embedding and Reranking Models

## Overview
This repository finetunes dense retrieval and reranking models on custom datasets using the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) framework.

**Primary focus**: Fine-tuning IR and reranker models. Evaluation tools are provided for model validation but are optional.

## Background
This work builds upon [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), a comprehensive framework for training and evaluating dense retrievers and rerankers.
We currently use the base finetuning setup. The original training scripts also support knowledge distillation via score injection in training data, which is a potential future improvement.

## Installation

### Quick Installation
For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

**For fine-tuning (recommended):**
```bash
pip install FlagEmbedding[finetune]
# or
pip install -r requirements-finetune.txt
```

**For inference only:**
```bash
pip install FlagEmbedding
```

**For fine-tuning + evaluation:**
```bash
pip install FlagEmbedding[finetune,eval]
```

**For development:**
```bash
pip install -e .[finetune,eval,dev]
# or
pip install -r requirements-dev.txt
```

For troubleshooting and alternative installation methods, see [INSTALLATION.md](INSTALLATION.md) and [QUICK_START.md](QUICK_START.md).

## Quickstart

### Using Pre-trained Models
See [QUICK_START.md](QUICK_START.md) for installation quick reference.

### Fine-tuning Models
1. Download datasets:
  ```bash
  python download_CodeConvo.py --split train
  python download_RFCAlign.py
  ```
2. Install fine-tuning dependencies:
  ```bash
  pip install -r requirements-finetune.txt
  ```
3. Run a finetuning script (examples below).

## Datasets
Training datasets:
- **CodeConvo**: https://huggingface.co/datasets/jiebi/CodeConvo
- **RFC-Align**: https://huggingface.co/datasets/jiebi/RFCAlign

### Download data
```
python download_CodeConvo.py --split train
python download_RFCAlign.py
```

See [FlagEmbedding/DATA_PATH_USAGE.md](FlagEmbedding/DATA_PATH_USAGE.md) for full usage and path rules.

## Training
Example finetuning scripts:
- [examples/finetune/embedder/decoder_only/ft_CodeConvo_decoder.sh](examples/finetune/embedder/decoder_only/ft_CodeConvo_decoder.sh)
- [examples/finetune/embedder/encoder_only/ft_CodeConvo_encoder.sh](examples/finetune/embedder/encoder_only/ft_CodeConvo_encoder.sh)
- [examples/finetune/embedder/decoder_only/ft_RFCAlign_verbose.sh](examples/finetune/embedder/decoder_only/ft_RFCAlign_verbose.sh)
- [examples/finetune/embedder/decoder_only/ft_RFCAlign_non-verbose.sh](examples/finetune/embedder/decoder_only/ft_RFCAlign_non-verbose.sh)

## Models
Trained retrieval models from CodeConvo:
- **IDs-C2I-DEC**: https://huggingface.co/jiebi/IDs-C2I-Dec
- **Kubernetes-C2I-DEC**: https://huggingface.co/jiebi/Kubernetes-C2I-Dec
- **SIGIR-C2I-DEC**: https://huggingface.co/jiebi/SIGIR-C2I-Dec
- **IDs-I2C-DEC**: https://huggingface.co/jiebi/IDs-I2C-Dec
- **Kubernetes-I2C-DEC**: https://huggingface.co/jiebi/Kubernetes-I2C-Dec
- **SIGIR-I2C-DEC**: https://huggingface.co/jiebi/SIGIR-I2C-Dec

Trained retrieval models from RFCAlign (V: verbose; N: non-verbose; D: decision; R: rationale):
- **RFC-DRAlign-QV**: https://huggingface.co/jiebi/RFC-DRAlign-QV
- **RFC-DRAlign-QL**: https://huggingface.co/jiebi/RFC-DRAlign-QL
- **RFC-DRAlign-LV**: https://huggingface.co/jiebi/RFC-DRAlign-LV
- **RFC-DRAlign-LN**: https://huggingface.co/jiebi/RFC-DRAlign-LN

## Evaluation
We evaluated the finetuned models using MTEB:
https://github.com/embeddings-benchmark/mteb

We also provide an evaluation wrapper here:
https://github.com/cheop-byeon/mteb-R2Gen

For evaluation setup and usage, see [examples/evaluation](examples/evaluation).

## Documentation
- [INSTALLATION.md](INSTALLATION.md) - Comprehensive installation guide with troubleshooting
- [QUICK_START.md](QUICK_START.md) - Quick reference and one-liners for common scenarios
- [DEPENDENCY_UPDATES.md](DEPENDENCY_UPDATES.md) - Details on dependency management updates
- [FlagEmbedding/DATA_PATH_USAGE.md](FlagEmbedding/DATA_PATH_USAGE.md) - Data path configuration

## References
- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)

### Citation
For data, use cases, and models, please refer to:
```
@article{bian2025automated,
  title={Automated insights into github collaboration dynamics},
  author={Bian, Jie and Arefev, Nikolay and M{\"u}hlh{\"a}user, Max and Welzl, Michael},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}

```
If you use this library, please cite the original FlagEmbedding work:
```
@misc{bge_m3,
  title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  author={Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  year={2023},
  eprint={2309.07597},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{llm_embedder,
  title={Retrieve Anything To Augment Large Language Models},
  author={Peitian Zhang and Shitao Xiao and Zheng Liu and Zhicheng Dou and Jian-Yun Nie},
  year={2023},
  eprint={2310.07554},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```