#!/usr/bin/env python
"""
CHG Example Training Script

This script runs Causal Head Gating analysis on a transformer model.
Designed to be run via Slurm on HPC clusters.

Usage:
    python run_chg_example.py --model meta-llama/Llama-3.2-3B-Instruct --dataset aba_abb
    python run_chg_example.py --model meta-llama/Llama-3.2-1B --num-updates 100  # Quick test
"""

# CRITICAL: HF_HOME must be set BEFORE importing transformers
# transformers/huggingface_hub cache the directory at import time
import os
if 'HF_HOME' not in os.environ:
    # Default for Princeton clusters
    default_hf = os.path.expanduser('~/scr/huggingface')
    if os.path.exists(default_hf):
        os.environ['HF_HOME'] = default_hf
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(default_hf, 'hub')

# Force offline mode - compute nodes have no internet access
# This prevents huggingface_hub from making ANY network requests
os.environ['HF_HUB_OFFLINE'] = '1'

import argparse
import torch
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from causal_head_gating import CHGTrainer
from causal_head_gating.data import MaskedSequenceDataset
from causal_head_gating.utils import to_long_df


def resolve_local_model_path(model_name: str) -> str:
    """
    Convert a HuggingFace model name to local cache path.

    When running offline, we need to pass the actual filesystem path to
    from_pretrained() to avoid transformers trying to contact HuggingFace.

    Args:
        model_name: HuggingFace model name like "meta-llama/Llama-3.2-1B"

    Returns:
        Local path to the cached model, or original name if not found
    """
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    hub_path = Path(hf_home) / 'hub'

    # HuggingFace caches models at: hub/models--{org}--{model}/snapshots/{hash}/
    cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache = hub_path / cache_name / 'snapshots'

    if model_cache.exists():
        # Get the most recent snapshot (there's usually only one)
        snapshots = list(model_cache.iterdir())
        if snapshots:
            local_path = str(snapshots[0])
            print(f"Resolved model to local path: {local_path}")
            return local_path

    print(f"Warning: Could not find local cache for {model_name}, using name directly")
    return model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Run CHG analysis on a transformer model")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aba_abb",
        help="Dataset name (e.g., aba_abb, math)"
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=500,
        help="Number of updates for unregularized phase"
    )
    parser.add_argument(
        "--num-reg-updates",
        type=int,
        default=500,
        help="Number of updates for each regularized phase"
    )
    parser.add_argument(
        "--gradient-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ../notebooks/config.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Print job info
    print("=" * 60)
    print("CHG Training Job")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: cuda:{args.device}")
    print(f"Num updates: {args.num_updates}")
    print(f"Num reg updates: {args.num_reg_updates}")

    # Print Slurm info if available
    if "SLURM_JOB_ID" in os.environ:
        print(f"Slurm Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"Slurm Node: {os.environ.get('SLURM_NODELIST', 'unknown')}")
    print("=" * 60)

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        # Default: look for config relative to this script
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "notebooks" / "config.yaml"

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent.resolve()
    # Expand ~ and resolve paths relative to config directory
    directories = {}
    for k, v in config['directories'].items():
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = config_dir / p
        directories[k] = p.resolve()

    # Set HuggingFace cache - MUST be set before importing transformers
    hf_home = str(directories['huggingface'])
    os.environ['HF_HOME'] = hf_home
    os.environ['TRANSFORMERS_CACHE'] = str(Path(hf_home) / 'hub')
    os.environ['HF_DATASETS_CACHE'] = str(Path(hf_home) / 'datasets')
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = directories['save'] / 'results' / args.dataset / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    print(f"CUDA device: {torch.cuda.get_device_name(args.device)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(args.device).total_memory / 1e9:.1f} GB")

    # Load tokenizer and model
    # Resolve to local path to avoid transformers trying to contact HuggingFace
    print(f"\nLoading model: {args.model}")
    local_model_path = resolve_local_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True).to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Load dataset
    dataset_path = directories['save'] / f'datasets/{args.dataset}/{args.model}/train.pt'
    print(f"\nLoading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please run the dataset preparation notebook first:\n"
            f"  notebooks/datasets/{args.dataset}.ipynb"
        )

    data = torch.load(dataset_path, weights_only=False)
    if 'text_tokens' in data:
        data['input_ids'] = data.pop('text_tokens')
    dataset = MaskedSequenceDataset(tokenizer.pad_token_id, **data).to(args.device)
    print(f"Dataset loaded. Samples: {len(dataset)}")

    # Train CHG
    print(f"\nStarting CHG training...")
    trainer = CHGTrainer(model, dataset, gradient_accum_steps=args.gradient_accum_steps)

    masks_list, metrics_list = [], []
    for mask, metric in trainer.fit(
        num_updates=args.num_updates,
        num_reg_updates=args.num_reg_updates,
        verbose=True
    ):
        masks_list.append(mask)
        metrics_list.append(metric)

    # Process results
    print(f"\nProcessing results...")
    masks = torch.stack(masks_list)
    masks = masks.view(3, -1, masks.shape[-2], masks.shape[-1])
    df_metrics = pd.DataFrame(metrics_list)
    df = to_long_df(masks, ['regularization', 'step', 'layer_idx', 'head_idx'])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    masks_path = output_dir / f"masks_{timestamp}.pt"
    torch.save({
        'masks': masks,
        'model_name': args.model,
        'dataset': args.dataset,
        'num_updates': args.num_updates,
        'num_reg_updates': args.num_reg_updates,
    }, masks_path)
    print(f"Saved masks to: {masks_path}")

    metrics_path = output_dir / f"metrics_{timestamp}.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    results_path = output_dir / f"results_{timestamp}.parquet"
    df.to_parquet(results_path)
    print(f"Saved results to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"\nMask shape: {masks.shape}")
    print(f"  [regularization, step, layer, head]")
    print(f"\nFinal metrics:")
    print(df_metrics.tail(3).to_string())

    # Summary statistics
    for i, reg in enumerate(['none', 'positive', 'negative']):
        final_mask = masks[i, -1].sigmoid()
        print(f"\n{reg.upper()} regularization final mask:")
        print(f"  Mean: {final_mask.mean():.3f}")
        print(f"  Active heads (>0.5): {(final_mask > 0.5).sum().item()}")
        print(f"  Inactive heads (<0.5): {(final_mask < 0.5).sum().item()}")


if __name__ == "__main__":
    main()
