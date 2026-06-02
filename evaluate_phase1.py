"""
Phase 1: Evaluate Transfer Matrix
=================================

Evaluates all trained Phase 1 models on all 3 representative codes
to build the 3x3 transfer matrix.

Usage:
    # Evaluate all models and save results
    python evaluate_phase1.py

    # Evaluate specific model
    python evaluate_phase1.py --model=Results_FECCT/BCH_N63_K45__xxx/best_model.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
import csv

from Codes import Get_Generator_and_Parity, EbN0_to_std, BER, FER, bin_to_sign, sign_to_bin
from Main_FECCT import Code, FECCT_Dataset, collate_single_code

# =============================================================================
# PHASE 1 CODES
# =============================================================================

CODES = {
    'BCH_63_45':   ('BCH', 63, 45),
    'POLAR_64_32': ('POLAR', 64, 32),
    'LDPC_49_24':  ('LDPC', 49, 24),
}


def evaluate_model_on_code(model, device, code, EbNo_range=[4, 5, 6],
                            min_errors=100, max_samples=1e6):
    """Evaluate a model on a single code."""
    model.eval()
    results = {}

    rate = code.k / code.n

    with torch.no_grad():
        for EbNo in EbNo_range:
            sigma = EbN0_to_std(EbNo, rate)
            dataset = FECCT_Dataset(code, [sigma], length=2048, zero_cw=False)
            loader = DataLoader(dataset, batch_size=512, collate_fn=collate_single_code)

            total_ber = total_fer = total_samples = 0
            total_frame_errors = 0

            while total_frame_errors < min_errors and total_samples < max_samples:
                batch = next(iter(loader))
                magnitude = batch['magnitude'].to(device)
                syndrome = batch['syndrome'].to(device)
                pc_matrix = batch['pc_matrix'].to(device)
                y = batch['channel_output'].to(device)
                x = batch['codeword'].to(device)

                z_mul = y * bin_to_sign(x)
                z_pred = model(magnitude, syndrome, pc_matrix)
                _, x_pred = model.loss(-z_pred, z_mul, y)

                ber = BER(x_pred, x)
                fer = FER(x_pred, x)

                total_ber += ber * x.shape[0]
                total_fer += fer * x.shape[0]
                total_samples += x.shape[0]
                total_frame_errors = int(total_fer)

            final_ber = total_ber / total_samples
            final_fer = total_fer / total_samples
            neg_ln_ber = -np.log(final_ber) if final_ber > 0 else float('inf')

            results[EbNo] = {
                'BER': float(final_ber),
                'FER': float(final_fer),
                'neg_ln_BER': float(neg_ln_ber),
                'samples': int(total_samples),
            }

    return results


def find_models(models_dir):
    """Find all trained models in directory."""
    models = []

    if not os.path.exists(models_dir):
        print(f"Directory not found: {models_dir}")
        return models

    for entry in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, entry)
        if not os.path.isdir(model_dir):
            continue

        best_model = os.path.join(model_dir, 'best_model.pt')
        if os.path.exists(best_model):
            # Parse model name (e.g., BCH_N63_K45__20260602_120000)
            parts = entry.split('__')[0]  # Remove timestamp

            models.append({
                'name': parts,
                'path': best_model,
                'dir': model_dir,
            })

    return models


def build_transfer_matrix(models_dir='Results_FECCT', output_file='transfer_matrix.csv', EbNo=5):
    """Build the 3x3 transfer matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all models
    models = find_models(models_dir)
    print(f"Found {len(models)} trained models")

    if len(models) == 0:
        print("No models found! Please train models first.")
        return None

    # Initialize code objects
    print("\nLoading code definitions...")
    code_objects = {}
    for code_name, (ct, n, k) in CODES.items():
        try:
            code_objects[code_name] = Code(ct, n, k)
            print(f"  Loaded {code_name}")
        except Exception as e:
            print(f"  ERROR loading {code_name}: {e}")

    # Build transfer matrix
    print(f"\nBuilding transfer matrix at EbNo={EbNo}dB...")

    results = defaultdict(dict)

    for model_info in models:
        model_name = model_info['name']
        print(f"\n[{model_name}]")

        try:
            model = torch.load(model_info['path'], map_location=device, weights_only=False)
            model.eval()
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        for code_name, code_obj in code_objects.items():
            try:
                eval_results = evaluate_model_on_code(
                    model, device, code_obj,
                    EbNo_range=[EbNo], min_errors=100, max_samples=1e6
                )
                neg_ln_ber = eval_results[EbNo]['neg_ln_BER']
                results[model_name][code_name] = neg_ln_ber
                print(f"  {code_name}: {neg_ln_ber:.2f}")
            except Exception as e:
                print(f"  ERROR on {code_name}: {e}")
                results[model_name][code_name] = None

    # Save results
    if output_file and results:
        save_results(results, output_file, list(CODES.keys()))

    return results


def save_results(results, output_file, codes):
    """Save transfer matrix to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Model'] + codes
        writer.writerow(header)

        # Data rows
        for model_name in sorted(results.keys()):
            row = [model_name]
            for code_name in codes:
                val = results[model_name].get(code_name)
                row.append(f"{val:.2f}" if val is not None else "")
            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")


def print_matrix(results, codes):
    """Print transfer matrix to console."""
    if not results:
        print("No results to display")
        return

    print("\n" + "="*70)
    print("TRANSFER MATRIX (-ln(BER) at 5dB)")
    print("="*70)

    # Header
    print(f"{'Model':<25}", end='')
    for code in codes:
        print(f"{code:<15}", end='')
    print()
    print("-"*70)

    # Data
    for model_name in sorted(results.keys()):
        print(f"{model_name:<25}", end='')
        for code_name in codes:
            val = results[model_name].get(code_name)
            if val is not None:
                print(f"{val:<15.2f}", end='')
            else:
                print(f"{'--':<15}", end='')
        print()

    print("="*70)
    print("\nDiagonal = specialist performance (same code)")
    print("Off-diagonal = transfer performance (different code)")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Evaluate Transfer Matrix')
    parser.add_argument('--models_dir', type=str, default='Results_FECCT',
                        help='Directory containing trained models')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to specific model to evaluate')
    parser.add_argument('--output', type=str, default='transfer_matrix_phase1.csv',
                        help='Output CSV file for results')
    parser.add_argument('--EbNo', type=int, default=5,
                        help='SNR point for evaluation (default: 5dB)')

    args = parser.parse_args()

    codes = list(CODES.keys())

    if args.model:
        # Single model evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = torch.load(args.model, map_location=device, weights_only=False)
        model.eval()

        print(f"Evaluating: {args.model}")
        for code_name in codes:
            ct, n, k = CODES[code_name]
            code_obj = Code(ct, n, k)
            results = evaluate_model_on_code(model, device, code_obj, EbNo_range=[args.EbNo])
            neg_ln_ber = results[args.EbNo]['neg_ln_BER']
            print(f"  {code_name}: -ln(BER) = {neg_ln_ber:.2f}")
    else:
        # Full matrix evaluation
        results = build_transfer_matrix(
            args.models_dir,
            output_file=args.output,
            EbNo=args.EbNo
        )
        print_matrix(results, codes)


if __name__ == '__main__':
    main()
