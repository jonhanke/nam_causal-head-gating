#!/bin/bash
#SBATCH --job-name=chg-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu                # <-- Change to your cluster's GPU partition
#SBATCH --account=YOUR_ACCOUNT         # <-- Change to your account/allocation
#SBATCH --output=chg_%j.out
#SBATCH --error=chg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# CHG Training Slurm Job Script
# ============================================================================
#
# Before first use:
#   1. Update --partition and --account above for your cluster
#   2. Update HF_HOME below to your HuggingFace cache location
#   3. Pre-download models on login node (compute nodes often lack internet):
#      huggingface-cli download meta-llama/Llama-3.2-1B
#
# Usage:
#   sbatch job_example.slurm                                  # Default model
#   sbatch --export=MODEL=meta-llama/Llama-3.2-1B job_example.slurm  # Smaller model
#   sbatch --gres=gpu:h100:1 job_example.slurm                # Request specific GPU
#
# ============================================================================

# Print job information
echo "=============================================="
echo "Slurm Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=============================================="

# Print environment
echo ""
echo "SLURM Environment Variables:"
printenv | grep -i slurm | sort
echo ""

# Default values (can be overridden via --export)
MODEL=${MODEL:-"meta-llama/Llama-3.2-3B-Instruct"}
DATASET=${DATASET:-"aba_abb"}
NUM_UPDATES=${NUM_UPDATES:-500}
NUM_REG_UPDATES=${NUM_REG_UPDATES:-500}

echo "=============================================="
echo "Job Parameters"
echo "=============================================="
echo "MODEL: $MODEL"
echo "DATASET: $DATASET"
echo "NUM_UPDATES: $NUM_UPDATES"
echo "NUM_REG_UPDATES: $NUM_REG_UPDATES"
echo "=============================================="

# Load modules
# Adjust module names based on your cluster's available modules
module purge
module load anaconda3/2024.6 2>/dev/null || module load anaconda3/2023.9 2>/dev/null || module load anaconda3

# Activate conda environment
# Option 1: Use a pre-created environment
# conda activate chg-env

# Option 2: Use the project's virtual environment (if using uv)
# source /path/to/nam_causal-head-gating/.venv/bin/activate

# Option 3: Create environment on-the-fly (first run only)
# This assumes the package is installed in a shared location or the current directory
# Use SLURM_SUBMIT_DIR since Slurm copies scripts to temp location
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Check if we're in the repo directory
if [ -f "$REPO_DIR/pyproject.toml" ]; then
    echo "Found repository at: $REPO_DIR"
    cd "$REPO_DIR"

    # If using uv and .venv exists
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "Installing package with pip..."
        pip install -e . --quiet
    fi
else
    echo "Repository not found. Assuming package is already installed."
fi

# Set HuggingFace cache BEFORE running Python (transformers caches this at import time)
# UPDATE THIS PATH for your cluster's scratch/cache location
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# Force offline mode - compute nodes have no internet
export HF_HUB_OFFLINE=1
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run the training script
echo "=============================================="
echo "Starting CHG Training"
echo "=============================================="

python "$SCRIPT_DIR/run_chg_example.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num-updates "$NUM_UPDATES" \
    --num-reg-updates "$NUM_REG_UPDATES" \
    --config "$SCRIPT_DIR/config.yaml"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job Complete"
echo "=============================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_CODE
