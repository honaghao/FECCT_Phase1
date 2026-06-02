#!/usr/bin/env python3
"""
Live Training Monitor with FECCT Paper Baselines
=================================================

Monitors training progress in real-time and compares against FECCT paper baselines.
Updates plot every 5 epochs.

Usage:
    # Monitor all active training runs
    python live_monitor.py

    # Monitor specific model
    python live_monitor.py --model=Results_FECCT/BCH_N63_K45__xxx

    # Save plots instead of displaying
    python live_monitor.py --save_dir=plots/

    # Change update interval
    python live_monitor.py --interval=10
"""

import os
import sys
import re
import time
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for live updates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# FECCT PAPER BASELINES (from ICLR 2024 paper, Table 1)
# =============================================================================

# Format: {code_name: {EbNo: -ln(BER)}}
FECCT_BASELINES = {
    'BCH_63_45': {
        4: 5.18,
        5: 7.32,
        6: 10.31,
    },
    'POLAR_64_32': {
        4: 5.88,
        5: 7.91,
        6: 10.76,
    },
    'LDPC_49_24': {
        4: 4.52,
        5: 6.38,
        6: 9.21,
    },
}

# Target epochs
TARGET_EPOCHS = 1000

# Colors for different codes
COLORS = {
    'BCH': '#1f77b4',      # Blue
    'POLAR': '#ff7f0e',    # Orange
    'LDPC': '#2ca02c',     # Green
}


def parse_training_log(log_path):
    """Parse training.log file to extract metrics."""
    if not os.path.exists(log_path):
        return None

    metrics = {
        'epochs': [],
        'loss': [],
        'ber': [],
        'fer': [],
        'neg_ln_ber': [],
        'eval_results': {},  # {epoch: {EbNo: neg_ln_ber}}
    }

    # Patterns to match
    epoch_pattern = re.compile(
        r'Epoch (\d+), Batch \d+/\d+:.*Loss=([\d.e+-]+).*BER=([\d.e+-]+).*FER=([\d.e+-]+)'
    )
    eval_pattern = re.compile(
        r'.*EbNo=(\d+)dB.*BER=([\d.e+-]+).*-ln\(BER\)=([\d.]+)'
    )
    eval_epoch_pattern = re.compile(r'=== Evaluation at epoch (\d+) ===')

    current_eval_epoch = None

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Match training metrics
                match = epoch_pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    ber = float(match.group(3))
                    fer = float(match.group(4))

                    # Only keep last entry per epoch
                    if epoch not in metrics['epochs'] or metrics['epochs'][-1] != epoch:
                        metrics['epochs'].append(epoch)
                        metrics['loss'].append(loss)
                        metrics['ber'].append(ber)
                        metrics['fer'].append(fer)
                        neg_ln_ber = -np.log(ber) if ber > 0 else 0
                        metrics['neg_ln_ber'].append(neg_ln_ber)
                    else:
                        # Update last entry
                        metrics['loss'][-1] = loss
                        metrics['ber'][-1] = ber
                        metrics['fer'][-1] = fer
                        neg_ln_ber = -np.log(ber) if ber > 0 else 0
                        metrics['neg_ln_ber'][-1] = neg_ln_ber

                # Match evaluation epoch marker
                eval_epoch_match = eval_epoch_pattern.search(line)
                if eval_epoch_match:
                    current_eval_epoch = int(eval_epoch_match.group(1))
                    if current_eval_epoch not in metrics['eval_results']:
                        metrics['eval_results'][current_eval_epoch] = {}

                # Match evaluation results
                eval_match = eval_pattern.search(line)
                if eval_match and current_eval_epoch is not None:
                    ebno = int(eval_match.group(1))
                    neg_ln_ber = float(eval_match.group(3))
                    metrics['eval_results'][current_eval_epoch][ebno] = neg_ln_ber

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None

    return metrics


def find_active_runs(results_dir='Results_FECCT'):
    """Find all training runs with log files."""
    runs = []

    if not os.path.exists(results_dir):
        return runs

    for entry in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, entry)
        log_path = os.path.join(run_dir, 'training.log')

        if os.path.exists(log_path):
            # Parse run name to get code type
            parts = entry.split('__')[0]
            code_type = parts.split('_')[0]  # BCH, POLAR, or LDPC

            runs.append({
                'name': entry,
                'short_name': parts,
                'code_type': code_type,
                'log_path': log_path,
                'dir': run_dir,
            })

    return runs


class LiveMonitor:
    """Live training monitor with FECCT baselines."""

    def __init__(self, results_dir='Results_FECCT', update_interval=5, save_dir=None):
        self.results_dir = results_dir
        self.update_interval = update_interval
        self.save_dir = save_dir
        self.last_update = {}

        # Setup figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('FECCT Training Monitor - Phase 1', fontsize=14, fontweight='bold')

        # Axes: [0,0]=Loss, [0,1]=BER, [1,0]=neg_ln_BER, [1,1]=Eval comparison
        self.ax_loss = self.axes[0, 0]
        self.ax_ber = self.axes[0, 1]
        self.ax_neg_ln_ber = self.axes[1, 0]
        self.ax_eval = self.axes[1, 1]

        self.setup_axes()

    def setup_axes(self):
        """Setup axis labels and titles."""
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlim(0, TARGET_EPOCHS)
        self.ax_loss.grid(True, alpha=0.3)

        self.ax_ber.set_xlabel('Epoch')
        self.ax_ber.set_ylabel('BER')
        self.ax_ber.set_title('Bit Error Rate (BER)')
        self.ax_ber.set_xlim(0, TARGET_EPOCHS)
        self.ax_ber.set_yscale('log')
        self.ax_ber.grid(True, alpha=0.3)

        self.ax_neg_ln_ber.set_xlabel('Epoch')
        self.ax_neg_ln_ber.set_ylabel('-ln(BER)')
        self.ax_neg_ln_ber.set_title('Training Performance (-ln(BER))')
        self.ax_neg_ln_ber.set_xlim(0, TARGET_EPOCHS)
        self.ax_neg_ln_ber.grid(True, alpha=0.3)

        self.ax_eval.set_xlabel('Code')
        self.ax_eval.set_ylabel('-ln(BER) at 5dB')
        self.ax_eval.set_title('Evaluation vs FECCT Paper Baseline')
        self.ax_eval.grid(True, alpha=0.3)

    def update(self, frame):
        """Update function called by animation."""
        runs = find_active_runs(self.results_dir)

        if not runs:
            return

        # Clear axes
        self.ax_loss.clear()
        self.ax_ber.clear()
        self.ax_neg_ln_ber.clear()
        self.ax_eval.clear()
        self.setup_axes()

        eval_data = {'names': [], 'current': [], 'baseline': [], 'colors': []}

        for run in runs:
            metrics = parse_training_log(run['log_path'])
            if metrics is None or len(metrics['epochs']) == 0:
                continue

            code_type = run['code_type']
            color = COLORS.get(code_type, 'gray')
            label = run['short_name']

            # Plot training curves
            epochs = metrics['epochs']

            self.ax_loss.plot(epochs, metrics['loss'], color=color, label=label, alpha=0.8)
            self.ax_ber.plot(epochs, metrics['ber'], color=color, label=label, alpha=0.8)
            self.ax_neg_ln_ber.plot(epochs, metrics['neg_ln_ber'], color=color, label=label, alpha=0.8)

            # Get latest evaluation at 5dB
            latest_eval = None
            for eval_epoch in sorted(metrics['eval_results'].keys(), reverse=True):
                if 5 in metrics['eval_results'][eval_epoch]:
                    latest_eval = metrics['eval_results'][eval_epoch][5]
                    break

            if latest_eval is not None:
                # Get baseline
                baseline_key = f"{code_type}_{run['short_name'].split('_')[1]}_{run['short_name'].split('_')[2]}"
                baseline = FECCT_BASELINES.get(baseline_key, {}).get(5, 0)

                eval_data['names'].append(label)
                eval_data['current'].append(latest_eval)
                eval_data['baseline'].append(baseline)
                eval_data['colors'].append(color)

        # Plot evaluation comparison
        if eval_data['names']:
            x = np.arange(len(eval_data['names']))
            width = 0.35

            bars1 = self.ax_eval.bar(x - width/2, eval_data['baseline'], width,
                                      label='FECCT Paper', color='lightgray', edgecolor='black')
            bars2 = self.ax_eval.bar(x + width/2, eval_data['current'], width,
                                      label='Current Training', color=eval_data['colors'], edgecolor='black')

            self.ax_eval.set_xticks(x)
            self.ax_eval.set_xticklabels(eval_data['names'], rotation=45, ha='right')
            self.ax_eval.legend()

            # Add value labels on bars
            for bar, val in zip(bars1, eval_data['baseline']):
                if val > 0:
                    self.ax_eval.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                      f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars2, eval_data['current']):
                if val > 0:
                    self.ax_eval.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                      f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # Add legends
        self.ax_loss.legend(loc='upper right', fontsize=8)
        self.ax_ber.legend(loc='upper right', fontsize=8)
        self.ax_neg_ln_ber.legend(loc='lower right', fontsize=8)

        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.fig.suptitle(f'FECCT Training Monitor - Phase 1\nLast update: {timestamp}',
                          fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Save if requested
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            self.fig.savefig(save_path, dpi=100, bbox_inches='tight')

    def run(self):
        """Start the live monitor."""
        print("="*60)
        print("FECCT Live Training Monitor")
        print("="*60)
        print(f"Monitoring: {self.results_dir}")
        print(f"Update interval: {self.update_interval} seconds")
        print("Press Ctrl+C to stop")
        print("="*60)

        # Create animation
        ani = FuncAnimation(
            self.fig,
            self.update,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            cache_frame_data=False
        )

        plt.show()


def print_status(results_dir='Results_FECCT'):
    """Print current training status to console."""
    runs = find_active_runs(results_dir)

    print("\n" + "="*70)
    print("TRAINING STATUS")
    print("="*70)

    if not runs:
        print("No active training runs found.")
        return

    for run in runs:
        metrics = parse_training_log(run['log_path'])
        if metrics is None:
            continue

        current_epoch = metrics['epochs'][-1] if metrics['epochs'] else 0
        current_ber = metrics['ber'][-1] if metrics['ber'] else 0
        current_neg_ln_ber = metrics['neg_ln_ber'][-1] if metrics['neg_ln_ber'] else 0
        progress = (current_epoch / TARGET_EPOCHS) * 100

        # Get baseline
        code_key = run['short_name'].replace('_N', '_').replace('_K', '_')
        parts = run['short_name'].split('_')
        if len(parts) >= 3:
            baseline_key = f"{parts[0]}_{parts[1][1:]}_{parts[2][1:]}"
            baseline = FECCT_BASELINES.get(baseline_key, {}).get(5, 0)
        else:
            baseline = 0

        print(f"\n{run['short_name']}")
        print(f"  Progress: {current_epoch}/{TARGET_EPOCHS} epochs ({progress:.1f}%)")
        print(f"  Current BER: {current_ber:.2e}")
        print(f"  Current -ln(BER): {current_neg_ln_ber:.2f}")
        if baseline > 0:
            print(f"  FECCT Baseline (5dB): {baseline:.2f}")
            gap = baseline - current_neg_ln_ber
            print(f"  Gap to baseline: {gap:+.2f} dB")

        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  [{bar}]")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Live Training Monitor with FECCT Baselines')
    parser.add_argument('--results_dir', type=str, default='Results_FECCT',
                        help='Directory containing training runs')
    parser.add_argument('--interval', type=int, default=5,
                        help='Update interval in seconds (default: 5)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plot snapshots')
    parser.add_argument('--status', action='store_true',
                        help='Print status and exit (no GUI)')
    parser.add_argument('--model', type=str, default=None,
                        help='Monitor specific model directory')

    args = parser.parse_args()

    if args.model:
        args.results_dir = os.path.dirname(args.model)

    if args.status:
        print_status(args.results_dir)
    else:
        monitor = LiveMonitor(
            results_dir=args.results_dir,
            update_interval=args.interval,
            save_dir=args.save_dir
        )
        try:
            monitor.run()
        except KeyboardInterrupt:
            print("\nMonitor stopped.")


if __name__ == '__main__':
    main()
