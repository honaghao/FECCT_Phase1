"""
Phase 1: Minimal 3x3 Transfer Matrix Training
==============================================

Trains 9 models (3 representative codes x 3 seeds) for the behavioral
transfer matrix study.

Configuration matches FECCT paper (ICLR 2024):
- d_model: 128
- N_dec: 6
- h: 8
- batch_size: 512
- epochs: 1000

Usage:
    # Train all 9 models sequentially
    python train_phase1.py

    # Train single code with single seed
    python train_phase1.py --code=BCH --seed=42

    # Dry run (show commands without executing)
    python train_phase1.py --dry_run

    # Disable wandb logging
    python train_phase1.py --no_wandb
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION (matches FECCT paper, epochs=1000)
# =============================================================================

CONFIG = {
    'd_model': 128,
    'N_dec': 6,
    'h': 8,
    'batch_size': 512,
    'epochs': 1000,
    'lr': 1e-4,
}

# Phase 1: 3 representative codes (one per family)
CODES = {
    'BCH':   ('BCH', 63, 45),
    'POLAR': ('POLAR', 64, 32),
    'LDPC':  ('LDPC', 49, 24),
}

# 3 seeds for statistical confidence
SEEDS = [42, 123, 456]

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def build_command(code_type, code_n, code_k, seed, wandb=True):
    """Build training command."""
    cmd = [sys.executable, 'Main_FECCT.py']

    # Config
    cmd.extend([f'--epochs={CONFIG["epochs"]}'])
    cmd.extend([f'--batch_size={CONFIG["batch_size"]}'])
    cmd.extend([f'--d_model={CONFIG["d_model"]}'])
    cmd.extend([f'--N_dec={CONFIG["N_dec"]}'])
    cmd.extend([f'--h={CONFIG["h"]}'])
    cmd.extend([f'--lr={CONFIG["lr"]}'])
    cmd.extend([f'--seed={seed}'])

    # Code
    cmd.extend([f'--code_type={code_type}'])
    cmd.extend([f'--code_n={code_n}'])
    cmd.extend([f'--code_k={code_k}'])

    # Wandb
    if wandb:
        cmd.append('--wandb')
        cmd.append('--wandb_project=FECCT-TransferMatrix-Phase1')

    return cmd


def run_training(cmd, dry_run=False):
    """Run a training command."""
    cmd_str = ' '.join(cmd)
    print(f"\n{'='*70}")
    print(f"Command: {cmd_str}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False


def train_all(codes=None, seeds=None, dry_run=False, wandb=True):
    """Train all Phase 1 models."""
    if codes is None:
        codes = list(CODES.keys())
    if seeds is None:
        seeds = SEEDS

    total = len(codes) * len(seeds)
    completed = 0

    print(f"\n{'#'*70}")
    print(f"# PHASE 1: MINIMAL 3x3 TRANSFER MATRIX")
    print(f"# Codes: {codes}")
    print(f"# Seeds: {seeds}")
    print(f"# Total: {total} models")
    print(f"# Config: d_model={CONFIG['d_model']}, batch={CONFIG['batch_size']}, epochs={CONFIG['epochs']}")
    print(f"{'#'*70}")

    start_time = datetime.now()

    for code_name in codes:
        code_type, n, k = CODES[code_name]
        for seed in seeds:
            completed += 1
            print(f"\n[{completed}/{total}] Training {code_name}({n},{k}) seed={seed}")
            print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            cmd = build_command(code_type, n, k, seed, wandb=wandb)
            success = run_training(cmd, dry_run=dry_run)

            if not success and not dry_run:
                print(f"\nStopping due to training failure")
                return False

    elapsed = datetime.now() - start_time
    print(f"\n{'#'*70}")
    print(f"# PHASE 1 COMPLETE")
    print(f"# Total time: {elapsed}")
    print(f"# Models trained: {total}")
    print(f"{'#'*70}")

    return True


def print_summary():
    """Print training plan summary."""
    print("\n" + "="*70)
    print("PHASE 1: MINIMAL 3x3 TRANSFER MATRIX")
    print("="*70)

    print("\n## Configuration (FECCT paper, epochs=1000)")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    print("\n## Representative Codes (1 per family)")
    for name, (ct, n, k) in CODES.items():
        print(f"  - {name}: {ct}({n},{k})")

    print("\n## Seeds")
    print(f"  {SEEDS}")

    print("\n## Total Models")
    print(f"  {len(CODES)} codes x {len(SEEDS)} seeds = {len(CODES) * len(SEEDS)} models")

    print("\n## Estimated Time (RTX 4070 16GB)")
    print(f"  ~12-18 hours per model")
    print(f"  ~4-5 days total for all 9 models")

    print("\n## Output")
    print("  Results will be saved to: Results_FECCT/")
    print("  Each model creates: best_model.pt, final_model.pt, training.log")

    print("\n" + "="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 1: Minimal 3x3 Transfer Matrix Training')
    parser.add_argument('--code', type=str, default=None, choices=['BCH', 'POLAR', 'LDPC'],
                        help='Train specific code only')
    parser.add_argument('--seed', type=int, default=None,
                        help='Use single seed (default: all 3 seeds)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show commands without running')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary only')

    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    codes = [args.code] if args.code else None
    seeds = [args.seed] if args.seed else None
    wandb = not args.no_wandb

    # Print summary first
    print_summary()

    # Confirm before starting
    if not args.dry_run:
        print("\nPress Enter to start training, or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    train_all(codes=codes, seeds=seeds, dry_run=args.dry_run, wandb=wandb)


if __name__ == '__main__':
    main()
