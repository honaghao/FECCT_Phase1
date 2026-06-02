# Phase 1: Minimal 3x3 Transfer Matrix

Training code for the behavioral transfer matrix study on FECCT (Foundation Error Correction Code Transformer).

## Overview

This phase trains **9 models** (3 representative codes × 3 seeds) to build the minimal transfer matrix:

| Code Family | Representative | Parameters |
|-------------|----------------|------------|
| BCH | BCH(63,45) | n=63, k=45, rate=0.714 |
| Polar | Polar(64,32) | n=64, k=32, rate=0.500 |
| LDPC | LDPC(49,24) | n=49, k=24, rate=0.490 |

## Configuration

Matches FECCT paper (ICLR 2024) with epochs reduced to 1000:

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| N_dec (layers) | 6 |
| h (heads) | 8 |
| batch_size | 512 |
| epochs | 1000 |
| learning_rate | 1e-4 (cosine decay) |

## Requirements

- **GPU**: RTX 4070 16GB (or equivalent with ≥12GB VRAM)
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (or 12.x)

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Phase_1

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Login to wandb for experiment tracking
wandb login
```

## Training

### Quick Start

```bash
# Make script executable and run
chmod +x run_training.sh
./run_training.sh
```

### Train All 9 Models

```bash
python train_phase1.py
```

This trains all 9 models sequentially. Estimated time: **4-5 days** on RTX 4070 16GB.

### Train in Background (Recommended for Long Runs)

```bash
# Using nohup (output saved to nohup.out)
nohup python train_phase1.py &

# Or using screen
screen -S fecct_training
python train_phase1.py
# Press Ctrl+A, then D to detach
# Reattach with: screen -r fecct_training

# Or using tmux
tmux new -s fecct_training
python train_phase1.py
# Press Ctrl+B, then D to detach
# Reattach with: tmux attach -t fecct_training
```

### Train Single Code

```bash
# Train BCH only (3 seeds)
python train_phase1.py --code=BCH

# Train Polar only (3 seeds)
python train_phase1.py --code=POLAR

# Train LDPC only (3 seeds)
python train_phase1.py --code=LDPC
```

### Train Single Model

```bash
# Train BCH with seed 42 only
python train_phase1.py --code=BCH --seed=42
```

### Dry Run (Preview Commands)

```bash
python train_phase1.py --dry_run
```

### Disable Wandb

```bash
python train_phase1.py --no_wandb
```

## Monitoring Training

### Live Plot Monitor (Recommended)

The live monitor shows training progress compared to FECCT paper baselines, updating every 5 seconds.

```bash
# Install tkinter (required for live plot)
sudo apt-get install python3-tk  # Ubuntu/Debian
# or
sudo dnf install python3-tkinter  # Fedora

# Start live monitor (in a separate terminal)
python live_monitor.py
```

**Features:**
- Training loss, BER, and -ln(BER) curves
- Comparison with FECCT paper baselines
- Auto-updates every 5 seconds
- Progress tracking for all active runs

```bash
# Monitor with custom interval (10 seconds)
python live_monitor.py --interval=10

# Save plot snapshots
python live_monitor.py --save_dir=plots/

# Console status only (no GUI)
python live_monitor.py --status
```

### Other Monitoring Methods

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training log
tail -f Results_FECCT/*/training.log

# Check wandb dashboard (if enabled)
# https://wandb.ai/your-username/FECCT-TransferMatrix-Phase1
```

## Output

Trained models are saved to `Results_FECCT/`:

```
Results_FECCT/
├── BCH_N63_K45__YYYYMMDD_HHMMSS/
│   ├── best_model.pt      # Best model checkpoint
│   ├── final_model.pt     # Final model at epoch 1000
│   └── training.log       # Training logs
├── POLAR_N64_K32__YYYYMMDD_HHMMSS/
│   └── ...
└── LDPC_N49_K24__YYYYMMDD_HHMMSS/
    └── ...
```

## Evaluation

After training, evaluate all models to build the transfer matrix:

```bash
python evaluate_phase1.py
```

This creates `transfer_matrix_phase1.csv` with the 3×3 performance matrix.

## Expected Results

The transfer matrix should show:
- **Diagonal** (specialist): Higher values (~6-8 dB)
- **Off-diagonal** (transfer): Lower values (~2-4 dB)
- **Transfer gap**: ~3-5 dB difference

Example output:
```
                         BCH_63_45    POLAR_64_32   LDPC_49_24
BCH_N63_K45              7.40         3.05          2.81
POLAR_N64_K32            3.12         6.85          2.95
LDPC_N49_K24             2.78         2.91          6.52
```

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors, reduce batch size:
```bash
# Edit train_phase1.py, change CONFIG['batch_size'] from 512 to 256
```

### Permission Denied
```bash
chmod +x run_training.sh
```

### Module Not Found
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

## Files

| File | Description |
|------|-------------|
| `train_phase1.py` | Main training orchestrator |
| `evaluate_phase1.py` | Evaluation & transfer matrix |
| `live_monitor.py` | Live training plot with FECCT baselines |
| `Main_FECCT.py` | Core training script |
| `FECCT_Model.py` | FECCT model architecture |
| `Codes.py` | Code utilities & ECC functions |
| `Codes_DB/` | Parity-check matrices |
| `run_training.sh` | Quick-start shell script |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{choukroun2024foundation,
  title={A Foundation Model for Error Correction Codes},
  author={Choukroun, Yoni and Wolf, Lior},
  booktitle={ICLR},
  year={2024}
}
```

## Contact

For questions, please contact: hhtran@cau.ac.kr
