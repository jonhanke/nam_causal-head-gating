# Running CHG on Slurm Clusters

This directory contains scripts for running Causal Head Gating (CHG) analysis on Slurm-managed HPC clusters.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jonhanke/nam_causal-head-gating.git
cd nam_causal-head-gating

# 2. Set up environment (first time only)
./slurm/setup_env.sh

# 3. Configure for your cluster
cd slurm
cp job_example.slurm job.slurm
cp config_example.yaml config.yaml
# Edit job.slurm: update --partition and --account
# Edit config.yaml: update huggingface cache path if needed

# 4. Download model (on login node with internet)
huggingface-cli download meta-llama/Llama-3.2-1B

# 5. Prepare dataset (see Dataset Preparation below)

# 6. Submit training job
sbatch job.slurm
```

## Files

| File | Description |
|------|-------------|
| `job_example.slurm` | Template Slurm job script (copy to `job.slurm`) |
| `config_example.yaml` | Template config (copy to `config.yaml`) |
| `run_chg_example.py` | Python training script (called by job.slurm) |
| `setup_env.sh` | Environment setup script |
| `README.md` | This file |

**Note:** `job.slurm` and `config.yaml` are gitignored since they contain cluster-specific settings.

## Job Configuration

### Default Settings

The default `job_example.slurm` requests:
- 1 node, 1 GPU
- 8 CPU cores (for data loading)
- 64 GB memory
- 1 hour runtime

### Customizing the Job

**Change model or dataset:**
```bash
sbatch --export=MODEL=meta-llama/Llama-3.2-1B,DATASET=aba_abb job.slurm
```

**Request specific GPU type (e.g., H100):**
```bash
sbatch --gres=gpu:h100:1 job.slurm
```

**Change runtime:**
```bash
sbatch --time=08:00:00 job.slurm
```

**Combine options:**
```bash
sbatch --gres=gpu:h100:1 --time=02:00:00 \
       --export=MODEL=meta-llama/Llama-3.2-1B job.slurm
```

### Memory Requirements by Model Size

| Model | GPU Memory | System Memory | Recommended GPU |
|-------|-----------|---------------|-----------------|
| Llama-3.2-1B | ~8 GB | 32 GB | Any modern GPU |
| Llama-3.2-3B | ~16 GB | 64 GB | A100/H100 |
| Llama-3.1-8B | ~32 GB | 96 GB | A100-80GB/H100 |

## Dataset Preparation

Before running CHG training, you need to prepare the tokenized dataset. This can be done interactively or as a separate job.

### Interactive (on login node or via salloc)

```bash
# Request an interactive session with GPU
salloc --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G --time=01:00:00

# Activate environment
source .venv/bin/activate  # or conda activate chg-env

# Run dataset preparation
cd notebooks/datasets
python -c "
import sys
sys.path.insert(0, '../..')
exec(open('aba_abb.ipynb').read())
"
# Or use jupyter/papermill
```

### As a Slurm Job

Create a dataset preparation job:
```bash
sbatch --job-name=chg-data --time=01:00:00 --gres=gpu:1 --mem=32G \
       --wrap="cd $PWD && source .venv/bin/activate && python -m jupyter nbconvert --execute notebooks/datasets/aba_abb.ipynb"
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output in real-time
tail -f chg_<jobid>.out

# Check job details
scontrol show job <jobid>

# Cancel a job
scancel <jobid>
```

## Output Files

Results are saved to `notebooks/output/results/<dataset>/<model>/`:
- `masks_<timestamp>.pt` - Trained mask tensors
- `metrics_<timestamp>.csv` - Training metrics
- `results_<timestamp>.parquet` - Long-format results for analysis

Slurm logs are saved in the `slurm/` directory:
- `chg_<jobid>.out` - Standard output
- `chg_<jobid>.err` - Standard error

## Troubleshooting

### "CUDA out of memory"
- Use a smaller model (`--export=MODEL=meta-llama/Llama-3.2-1B`)
- Request a GPU with more memory (`--gres=gpu:a100:1` or `--gres=gpu:h100:1`)
- Increase gradient accumulation (`--export=GRAD_ACCUM=4`)

### "Dataset not found"
- Run the dataset preparation notebook first
- Check that `config.yaml` has correct paths
- Verify the model name matches exactly

### "Module not found"
- Check that the conda environment is activated
- Run `./slurm/setup_env.sh` to reinstall

### "Cannot resolve huggingface.co" / Network errors
- Compute nodes typically have no internet access
- Pre-download models on the login node: `huggingface-cli download <model>`
- The job script sets `HF_HUB_OFFLINE=1` to prevent network attempts

### Job pending too long
- Check cluster load: `squeue -p <partition>`
- Try a different partition or reduce resource requests

## Contact

For package issues: https://github.com/jonhanke/nam_causal-head-gating/issues
