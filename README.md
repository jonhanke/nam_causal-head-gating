# Causal Head Gating (CHG)

> Reproducible research code for interpreting attention head roles in transformer models via causal gating.
> Built with **Python 3.10+**, **PyTorch**, and **HuggingFace Transformers** for a clean, portable workflow.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/nam-causal-head-gating.svg)](https://pypi.org/project/nam-causal-head-gating/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [HPC Deployment](#hpc-deployment)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Citation](#citation)
- [License](#license)


## Overview

**Causal Head Gating (CHG)** is a scalable method for understanding what attention heads do in transformer models. This package accompanies the paper:

> **[Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](https://arxiv.org/abs/2505.13737)**
> Andrew Nam, Henry Conklin, Yukang Yang, Thomas Griffiths, Jonathan Cohen, Sarah-Jane Leslie
> NeurIPS 2025

Unlike traditional interpretability approaches that require hypothesis-driven analysis or specific prompt templates, CHG:

- **Learns which heads matter** by training soft gates over attention heads
- **Identifies necessary heads** (minimal set required for a task)
- **Identifies sufficient heads** (maximal set that enables performance)
- **Classifies head roles** as facilitating, interfering, or irrelevant
- **Works on any task** using standard next-token prediction

Our target audience is:

- **Interpretability researchers** who want a scalable, task-agnostic method for understanding attention head roles
- **ML engineers** who want to identify which heads are critical for specific capabilities
- **Researchers** interested in mechanistic interpretability without requiring manual circuit analysis


## Installation

To install the package, you can use [uv](https://docs.astral.sh/uv/) (recommended), [pip](https://pip.pypa.io/), or [conda](https://anaconda.org/anaconda/conda).

### Install from PyPI

```bash
pip install nam-causal-head-gating
```

### Install from Source with uv (recommended)

```bash
git clone https://github.com/jonhanke/nam_causal-head-gating
cd nam_causal-head-gating
make sync  # Uses uv.lock for reproducible installs
```

### Install from Source with pip

```bash
git clone https://github.com/jonhanke/nam_causal-head-gating
cd nam_causal-head-gating
pip install -e .
```

### Install with Conda

```bash
git clone https://github.com/jonhanke/nam_causal-head-gating
cd nam_causal-head-gating
conda env create -f environment.yml
conda activate causal-head-gating
pip install -e .
```

### Optional Dependencies

```bash
# For visualization
pip install nam-causal-head-gating[viz]

# For HuggingFace datasets integration
pip install nam-causal-head-gating[datasets]

# For Jupyter notebooks
pip install nam-causal-head-gating[notebooks]

# For development
pip install nam-causal-head-gating[dev]

# Everything
pip install nam-causal-head-gating[all]
```

### Running Jupyter Notebooks

After installation, you can run the example notebooks:

```bash
# Using make (recommended)
make notebook    # Starts Jupyter Notebook
make lab         # Starts JupyterLab

# Or manually
jupyter notebook --notebook-dir=notebooks
```

If using a virtual environment, ensure the kernel is registered:

```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=causal-head-gating --display-name="Python (CHG)"
```

Then select "Python (CHG)" as the kernel when opening notebooks.


## Quick Start

```python
from causal_head_gating import CHGAnalyzer

# Load any HuggingFace model
analyzer = CHGAnalyzer.from_pretrained("meta-llama/Llama-3.2-1B")

# Analyze on your data
results = analyzer.fit(
    texts=["The capital of France is", "2 + 2 equals"],
    targets=["Paris", "4"],
)

# Get insights
necessary = results.necessary_heads()
print(f"Found {len(necessary)} necessary heads")

# View head taxonomy
taxonomy = results.head_taxonomy()
print(taxonomy)
```


## Project Structure

```text
nam_causal-head-gating/
├── pyproject.toml                    # Package configuration
├── uv.lock                           # Reproducible installs (uv)
├── environment.yml                   # Conda environment
├── Makefile                          # Convenience commands
├── README.md
├── LICENSE
├── notebooks/
│   ├── config.yaml                   # Local paths configuration
│   ├── chg_example.ipynb             # CHG training example
│   └── datasets/
│       ├── aba_abb.ipynb             # ABA/ABB dataset preparation
│       └── math.ipynb                # Math dataset preparation
├── slurm/
│   ├── job_example.slurm             # Slurm job template
│   ├── config_example.yaml           # Cluster config template
│   ├── run_chg_example.py            # Training script for HPC
│   └── setup_env.sh                  # Environment setup
├── src/
│   └── causal_head_gating/
│       ├── __init__.py               # Public API exports
│       ├── api.py                    # High-level CHGAnalyzer API
│       ├── core/
│       │   ├── chg.py                # Core CHG class (hook-based gating)
│       │   └── trainer.py            # Three-stage training pipeline
│       ├── data/
│       │   ├── datasets.py           # MaskedSequenceDataset, TensorDict
│       │   ├── formatters.py         # Few-shot prompt formatting
│       │   └── tokenization.py       # PromptTokenizer utilities
│       ├── models/
│       │   └── adapters.py           # Model architecture adapters
│       ├── analysis/
│       │   └── masks.py              # Mask analysis utilities
│       └── utils/
│           ├── helpers.py            # to_long_df and other helpers
│           └── tensor_dict.py        # TensorDict implementation
└── tests/
    └── test_imports.py               # Import tests
```

**Note:** Dataset files (e.g., `aba_abb.tsv`) are hosted on [HuggingFace Hub](https://huggingface.co/datasets/jonhanke-nam/nam-causal-head-gating) and downloaded automatically on first use.


## Core Concepts

### Three-Stage Training

CHG training proceeds in three stages:

1. **Unregularized**: Learn task-relevant masks without sparsity bias
2. **Positive L1**: Find minimal *necessary* heads (pushes masks → 0)
3. **Negative L1**: Find maximal *sufficient* heads (pushes masks → 1)

### Head Taxonomy

Based on necessary and sufficient masks, heads are classified as:

| Classification | Necessary | Sufficient | Interpretation |
|----------------|-----------|------------|----------------|
| **Facilitating** | High | High | Helps the task |
| **Interfering** | Low | High | Hurts the task |
| **Irrelevant** | Low | Low | No effect |

### Loss Masking

CHG uses loss masks to specify which tokens to supervise:
- For **pattern tasks**: mask only the final prediction token
- For **generation tasks**: mask the entire target sequence
- For **few-shot learning**: mask only target tokens, not in-context examples


## Usage Examples

### Pattern Task (Finding Induction Heads)

```python
# ABA pattern tests for induction/copying behavior
results = analyzer.fit_pattern_task("aba", num_samples=5000)
print(results.necessary_heads())  # These are likely induction heads
```

### HuggingFace Dataset

```python
results = analyzer.fit_huggingface(
    dataset_name="openai/gsm8k",
    input_column="question",
    target_column="answer",
    max_samples=1000,
)
```

### Custom Dataset

```python
from causal_head_gating import CHGDataset

dataset = CHGDataset.from_texts(
    texts=my_prompts,
    targets=my_targets,
    tokenizer=analyzer.tokenizer,
)
results = analyzer.fit_dataset(dataset)
```

### Loading the ABA/ABB Dataset

```python
from causal_head_gating import CHGDataset
from causal_head_gating.data import get_aba_abb_path

# Download from HuggingFace (cached after first use)
data_path = get_aba_abb_path()

# Load with last-token-only supervision
dataset = CHGDataset.from_tsv(
    str(data_path),
    tokenizer=tokenizer,
    prompt_column="prompt",
    target_column="target",
    last_token_only=True,  # Only supervise final token
)
```

### Few-Shot Math Dataset

```python
from causal_head_gating.data import load_math_dataset

# Full workflow: load, filter, create few-shot prompts
df_prompts, input_ids, loss_masks = load_math_dataset(
    tokenizer=tokenizer,
    num_examples=50,        # Few-shot examples
    num_train=50000,        # Training samples
    num_validation=5000,    # Validation samples
)
```

### Advanced: Low-Level API

```python
from causal_head_gating import CHG, CHGTrainer

# Wrap your model
chg = CHG(model)

# Create and use masks directly
masks = chg.create_masks(num_masks=10)
logp = chg.get_logp(masks.sigmoid(), input_ids, loss_masks)

# Or use the trainer for full control
trainer = CHGTrainer(
    model, dataset,
    num_masks=10,
    l1_weight=0.15,
    gradient_accum_steps=4,
)

for mask, metrics in trainer.fit(num_updates=500, num_reg_updates=500):
    print(f"Stage: {metrics['regularization']}, NLL: {metrics['nll']:.3f}")
```


## HPC Deployment

For running CHG on Slurm-managed HPC clusters, we provide ready-to-use job scripts in the `slurm/` directory.

### Quick Start

```bash
cd slurm

# 1. Configure for your cluster
cp job_example.slurm job.slurm
cp config_example.yaml config.yaml
# Edit job.slurm: set --partition and --account for your cluster
# Edit config.yaml: set huggingface cache path

# 2. Download model on login node (compute nodes often lack internet)
huggingface-cli download meta-llama/Llama-3.2-1B

# 3. Submit job
sbatch job.slurm
```

### Features

- **Offline-ready**: Handles clusters where compute nodes have no internet access
- **Configurable**: Easy customization of model, dataset, and training parameters
- **GPU-optimized**: Mixed precision training, tested on A100/H100/H200

See [`slurm/README.md`](slurm/README.md) for detailed setup instructions, troubleshooting, and cluster-specific configuration.


## Supported Models

CHG supports most HuggingFace transformer architectures:

| Architecture | Models |
|--------------|--------|
| **Llama** | Llama-2, Llama-3, Llama-3.1, Llama-3.2 |
| **Mistral** | Mistral, Mixtral |
| **GPT-2** | GPT-2 (all sizes) |
| **GPT-NeoX** | Pythia |
| **Falcon** | Falcon |
| **Qwen** | Qwen2, Qwen2.5 |
| **Gemma** | Gemma, Gemma-2 |
| **Phi** | Phi, Phi-3 |

To add support for other architectures, subclass `ModelAdapter`.


## API Reference

### CHGAnalyzer

High-level interface for CHG analysis.

```python
CHGAnalyzer.from_pretrained(model_name, device=None, torch_dtype=None)
analyzer.fit(texts, targets, num_masks=10, num_updates=500, ...)
analyzer.fit_pattern_task(pattern="aba", num_samples=10000, ...)
analyzer.fit_huggingface(dataset_name, input_column, target_column, ...)
analyzer.fit_dataset(dataset, ...)
```

### CHGResults

Container for analysis results.

```python
results.necessary_heads(threshold=0.5)  # List of (layer, head) tuples
results.sufficient_heads(threshold=0.5)
results.head_taxonomy()  # DataFrame with classifications
results.summary()  # Dict with statistics
results.to_dataframe()  # Long-format DataFrame
results.save(path) / CHGResults.load(path)
```

### CHGDataset

Factory for creating training datasets.

```python
CHGDataset.from_texts(texts, targets, tokenizer)
CHGDataset.from_tokens(input_ids, loss_masks, pad_token_id)
CHGDataset.from_huggingface(dataset_name, tokenizer, input_column, target_column)
CHGDataset.from_pattern_task(pattern, tokenizer, num_samples)
CHGDataset.from_tsv(path, tokenizer, prompt_column, target_column, last_token_only=False)
```

### Utility Functions

```python
from causal_head_gating.utils import to_long_df
from causal_head_gating.data import load_math_dataset, create_fewshot_dataset, get_aba_abb_path
```


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{nam2025causal,
  title={Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers},
  author={Nam, Andrew and Conklin, Henry and Yang, Yukang and Griffiths, Thomas and Cohen, Jonathan and Leslie, Sarah-Jane},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```


## License

MIT License. See [LICENSE](LICENSE) for details.
