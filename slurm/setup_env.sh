#!/bin/bash
# ============================================================================
# Setup script for CHG environment on Slurm clusters
# ============================================================================
#
# Usage:
#   ./slurm/setup_env.sh           # Use uv (recommended)
#   ./slurm/setup_env.sh --conda   # Use conda instead
#
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "CHG Environment Setup"
echo "=============================================="
echo "Repository: $REPO_DIR"
echo "=============================================="

cd "$REPO_DIR"

# Parse arguments
USE_CONDA=false
if [[ "$1" == "--conda" ]]; then
    USE_CONDA=true
fi

if $USE_CONDA; then
    echo ""
    echo "Setting up with Conda..."
    echo ""

    # Load anaconda module
    module purge
    module load anaconda3/2024.6 2>/dev/null || \
    module load anaconda3/2023.9 2>/dev/null || \
    module load anaconda3

    # Create conda environment
    ENV_NAME="chg-env"

    if conda env list | grep -q "^$ENV_NAME "; then
        echo "Environment '$ENV_NAME' already exists. Updating..."
        conda activate $ENV_NAME
        pip install -e ".[all]" --upgrade
    else
        echo "Creating new conda environment: $ENV_NAME"
        conda create -n $ENV_NAME python=3.11 -y
        conda activate $ENV_NAME

        # Install PyTorch with CUDA support
        # Adjust CUDA version as needed for your cluster
        pip install torch --index-url https://download.pytorch.org/whl/cu121

        # Install the package with all dependencies
        pip install -e ".[all]"
    fi

    echo ""
    echo "Conda environment ready!"
    echo "Activate with: conda activate $ENV_NAME"

else
    echo ""
    echo "Setting up with uv..."
    echo ""

    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Create virtual environment and install
    if [ -d ".venv" ]; then
        echo "Virtual environment exists. Syncing dependencies..."
        uv sync
    else
        echo "Creating virtual environment..."
        uv venv
        uv sync
    fi

    echo ""
    echo "Virtual environment ready!"
    echo "Activate with: source .venv/bin/activate"
fi

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

if $USE_CONDA; then
    python -c "import causal_head_gating; print(f'causal_head_gating imported successfully')"
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
else
    uv run python -c "import causal_head_gating; print(f'causal_head_gating imported successfully')"
    uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Update notebooks/config.yaml with your paths"
echo "  2. Prepare dataset: run notebooks/datasets/aba_abb.ipynb"
echo "  3. Submit job: cd slurm && sbatch job.slurm"
echo ""
