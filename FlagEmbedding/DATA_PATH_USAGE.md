# Dataset Download Usage

This guide covers downloading datasets used for fine-tuning embedding and reranking models.

---

## CodeConvo Dataset

### Overview
The `download_CodeConvo.py` script downloads the **entire** CodeConvo dataset from Hugging Face and provides path resolution for specific data splits.

### Dataset Structure

#### Train Data
```
train/{direction}/
```
- **direction**: `i2c` (Issue-to-Code) or `c2i` (Code-to-Issue)

#### Dev/Test Data
```
{repo}/{direction}/{split}/
```
- **repo**: Repository name (`ids`, `ids-supp`, `swe`, `kubernetes`)
- **direction**: `i2c` (Issue-to-Code) or `c2i` (Code-to-Issue)
- **split**: `dev` or `test`

### Usage Examples

#### Download entire dataset
```bash
python download_CodeConvo.py
```

#### Get train data path (defaults to c2i direction)
```bash
python download_CodeConvo.py --split train
# → Returns: ./dataset/CodeConvo/train/c2i
```

#### Get train data path with specific direction
```bash
python download_CodeConvo.py --split train --direction i2c
# → Returns: ./dataset/CodeConvo/train/i2c
```

#### Get dev/test data path (requires repo and direction)
```bash
python download_CodeConvo.py --split test --repo kubernetes --direction i2c
# → Returns: ./dataset/CodeConvo/kubernetes/i2c/test
```

### Command-Line Arguments

| Argument | Description | Constraints |
|----------|-------------|-------------|
| `--split` | Data split (`train`, `dev`, `test`) | Optional |
| `--direction` | Retrieval direction (`i2c`, `c2i`) | For train: optional (defaults to `c2i`)<br>For dev/test: **required** |
| `--repo` | Repository name (`ids`, `ids-supp`, `swe`, `kubernetes`) | For train: **not allowed**<br>For dev/test: **required** |
| `--no-download` | Skip download, only resolve path | Optional flag |

### Important Notes

1. **The script always downloads the entire dataset** - no partial downloads
2. **Train split**:
   - Only accepts `--direction` parameter (defaults to `c2i`)
   - Cannot specify `--repo` (error will be shown)
3. **Dev/Test splits**:
   - Must specify both `--repo` and `--direction`
   - Path validation checks if the requested path exists

### Output Location
```
./dataset/CodeConvo/
```

The script preserves the original folder structure and creates a completion marker (`.download_complete`) to avoid re-downloading.

---

## RFCAlign Dataset

### Overview
The `download_RFCAlign.py` script downloads the **entire** RFCAlign dataset from Hugging Face for fine-tuning retrieval models.

### Dataset Structure
```
{model_type}_non-verbose/
{model_type}_verbose/
```
- **model_type**: Model architecture variant (e.g., `qwen`, `llama`)
- **verbose vs non-verbose**: Different prompt formatting styles

### Usage

#### Download entire dataset
```bash
python download_RFCAlign.py
```

The script will:
1. Examine the repository structure
2. Download all dataset files
3. Display the downloaded structure

### Dataset Structure Example
```
./dataset/RFCAlign/
├── qwen_non-verbose/
│   ├── data files...
├── qwen_verbose/
│   ├── data files...
├── llama_non-verbose/
│   ├── data files...
└── llama_verbose/
    ├── data files...
```

### Using in Training Scripts
In your training script, specify the path based on your model type:
```bash
MODEL="qwen"  # or "llama"
train_data="./dataset/RFCAlign/${MODEL}_non-verbose"
```

### Output Location
```
./dataset/RFCAlign/
```

The script creates a completion marker (`.download_complete`) to avoid re-downloading.
