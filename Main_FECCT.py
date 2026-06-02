"""
FECCT Training for Transfer Analysis Paper (Option 5)
Multi-code training with mixture support for BCH and Polar codes.

Usage:
    # Single code training (baseline)
    python Main_FECCT.py --code_type=BCH --code_n=63 --code_k=45

    # Mix-A: BCH codes only
    python Main_FECCT.py --mixture=A

    # Mix-B: Polar codes only
    python Main_FECCT.py --mixture=B

    # Mix-C: BCH + Polar heterogeneous
    python Main_FECCT.py --mixture=C

    # Zero-shot evaluation on held-out code
    python Main_FECCT.py --eval_only --checkpoint=path/to/model --code_type=BCH --code_n=127 --code_k=99
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils import data
from datetime import datetime
import logging
import numpy as np
import torch
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

from Codes import Get_Generator_and_Parity, EbN0_to_std, BER, FER, bin_to_sign, sign_to_bin
from FECCT_Model import FECCT_Decoder

# Wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Mixture definitions for behavioral transfer matrix experiments
MIXTURES = {
    # Homogeneous mixtures (within-family)
    'A': [  # BCH codes only
        ('BCH', 31, 16),
        ('BCH', 63, 45),
        ('BCH', 127, 99),
    ],
    'B': [  # Polar codes only
        ('POLAR', 64, 32),
        ('POLAR', 64, 48),
        ('POLAR', 128, 64),
    ],
    'L': [  # LDPC codes only
        ('LDPC', 49, 24),
        ('LDPC', 121, 60),
        ('LDPC', 121, 70),
    ],
    # Heterogeneous mixtures (cross-family)
    'C': [  # BCH + Polar
        ('BCH', 31, 16),
        ('BCH', 63, 45),
        ('POLAR', 64, 32),
        ('POLAR', 128, 64),
    ],
    'D': [  # BCH + LDPC
        ('BCH', 31, 16),
        ('BCH', 63, 45),
        ('LDPC', 49, 24),
        ('LDPC', 121, 60),
    ],
    'E': [  # Polar + LDPC
        ('POLAR', 64, 32),
        ('POLAR', 128, 64),
        ('LDPC', 49, 24),
        ('LDPC', 121, 60),
    ],
    'F': [  # All three families (full heterogeneous)
        ('BCH', 63, 45),
        ('POLAR', 64, 32),
        ('LDPC', 49, 24),
    ],
}

# Held-out codes for zero-shot evaluation
HELD_OUT_CODES = [
    ('BCH', 63, 36),
    ('BCH', 63, 51),
    ('POLAR', 128, 86),
    ('POLAR', 128, 96),
    ('LDPC', 121, 80),
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Code:
    """Container for code parameters and matrices."""
    def __init__(self, code_type, n, k):
        self.code_type = code_type
        self.n = n
        self.k = k
        G, H = Get_Generator_and_Parity(self, standard_form=False)
        # Keep on CPU - will be moved to GPU during training
        # Store transposed to match original ECCT convention
        self.generator_matrix = torch.from_numpy(G).float().transpose(0, 1)
        self.pc_matrix = torch.from_numpy(H).float()

    def __repr__(self):
        return f"{self.code_type}({self.n},{self.k})"


class FECCT_Dataset(data.Dataset):
    """Dataset for FECCT training on a single code."""

    def __init__(self, code, sigma_list, length, zero_cw=True):
        self.code = code
        self.sigma_list = sigma_list
        self.len = length
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)
        self.zero_cw = zero_cw

        if zero_cw:
            self.zero_word = torch.zeros((code.k,)).long()
            self.zero_codeword = torch.zeros((code.n,)).long()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw:
            m = self.zero_word
            x = self.zero_codeword
        else:
            m = torch.randint(0, 2, (self.code.k,))
            x = torch.matmul(m.float(), self.generator_matrix.float()).long() % 2

        sigma = random.choice(self.sigma_list)
        z = torch.randn(self.code.n) * sigma
        y = bin_to_sign(x.float()) + z

        magnitude = torch.abs(y)
        syndrome = torch.matmul(
            sign_to_bin(torch.sign(y)).long(),
            self.pc_matrix.long()
        ) % 2
        syndrome = bin_to_sign(syndrome.float())

        return {
            'message': m.float(),
            'codeword': x.float(),
            'noise': z.float(),
            'channel_output': y.float(),
            'magnitude': magnitude.float(),
            'syndrome': syndrome.float(),
            'pc_matrix': self.code.pc_matrix,  # Include for FECCT attention
            'code_info': (self.code.code_type, self.code.n, self.code.k),
        }


class MultiCodeDataset(data.Dataset):
    """Dataset that samples from multiple codes (for mixture training)."""

    def __init__(self, codes, sigma_list, samples_per_code, zero_cw=True):
        self.datasets = [
            FECCT_Dataset(code, sigma_list, samples_per_code, zero_cw)
            for code in codes
        ]
        self.total_len = len(self.datasets) * samples_per_code

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        dataset_idx = index % len(self.datasets)
        sample_idx = index // len(self.datasets)
        return self.datasets[dataset_idx][sample_idx]


def collate_single_code(batch):
    """Collate function for single-code batches (all same PC matrix)."""
    return {
        'message': torch.stack([b['message'] for b in batch]),
        'codeword': torch.stack([b['codeword'] for b in batch]),
        'noise': torch.stack([b['noise'] for b in batch]),
        'channel_output': torch.stack([b['channel_output'] for b in batch]),
        'magnitude': torch.stack([b['magnitude'] for b in batch]),
        'syndrome': torch.stack([b['syndrome'] for b in batch]),
        'pc_matrix': batch[0]['pc_matrix'],  # Same for all in batch
        'code_info': batch[0]['code_info'],
    }


def train_epoch(model, device, train_loader, optimizer, epoch, lr, use_wandb=False):
    """Train for one epoch."""
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()

    for batch_idx, batch in enumerate(train_loader):
        magnitude = batch['magnitude'].to(device)
        syndrome = batch['syndrome'].to(device)
        pc_matrix = batch['pc_matrix'].to(device)
        y = batch['channel_output'].to(device)
        x = batch['codeword'].to(device)

        z_mul = y * bin_to_sign(x)

        z_pred = model(magnitude, syndrome, pc_matrix)
        loss, x_pred = model.loss(-z_pred, z_mul, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        ber = BER(x_pred, x)
        fer = FER(x_pred, x)

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]

        if (batch_idx + 1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: '
                f'LR={lr:.2e}, Loss={cum_loss / cum_samples:.2e}, '
                f'BER={cum_ber / cum_samples:.2e}, FER={cum_fer / cum_samples:.2e}'
            )

    epoch_loss = cum_loss / cum_samples
    epoch_ber = cum_ber / cum_samples
    epoch_fer = cum_fer / cum_samples
    epoch_time = time.time() - t

    # Compute -ln(BER) metric used in FECCT paper
    neg_ln_ber = -np.log(epoch_ber) if epoch_ber > 0 else 0

    logging.info(f'Epoch {epoch} Train Time: {epoch_time:.1f}s\n')

    # Log to Wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'epoch': epoch,
            'train/loss': epoch_loss,
            'train/ber': epoch_ber,
            'train/fer': epoch_fer,
            'train/neg_ln_ber': neg_ln_ber,
            'train/lr': lr,
            'train/epoch_time': epoch_time,
        })

    return epoch_loss, epoch_ber, epoch_fer


def evaluate(model, device, code, EbNo_range, min_errors=100, max_samples=1e7, use_wandb=False, epoch=None):
    """Evaluate model on a single code at multiple SNR points."""
    model.eval()
    results = {'EbNo': [], 'BER': [], 'FER': [], 'neg_ln_BER': [], 'samples': []}

    rate = code.k / code.n

    with torch.no_grad():
        for EbNo in EbNo_range:
            sigma = EbN0_to_std(EbNo, rate)
            dataset = FECCT_Dataset(code, [sigma], length=2048, zero_cw=False)
            loader = DataLoader(dataset, batch_size=2048, collate_fn=collate_single_code)

            total_ber = total_fer = total_samples = 0
            total_errors = 0

            while total_errors < min_errors and total_samples < max_samples:
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
                total_errors = total_fer  # Use frame errors for stopping

            final_ber = total_ber / total_samples
            final_fer = total_fer / total_samples
            neg_ln_ber = -np.log(final_ber) if final_ber > 0 else 0

            results['EbNo'].append(EbNo)
            results['BER'].append(final_ber)
            results['FER'].append(final_fer)
            results['neg_ln_BER'].append(neg_ln_ber)
            results['samples'].append(total_samples)

            logging.info(f'{code}: EbNo={EbNo}dB, BER={final_ber:.2e}, '
                        f'FER={final_fer:.2e}, -ln(BER)={neg_ln_ber:.2f}, samples={int(total_samples)}')

            # Log to Wandb
            if use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    f'eval/{code}/EbNo_{EbNo}dB/BER': final_ber,
                    f'eval/{code}/EbNo_{EbNo}dB/FER': final_fer,
                    f'eval/{code}/EbNo_{EbNo}dB/neg_ln_BER': neg_ln_ber,
                }
                if epoch is not None:
                    log_dict['epoch'] = epoch
                wandb.log(log_dict)

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Initialize Wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        # Create run name
        if args.mixture:
            run_name = f'Mix{args.mixture}_d{args.d_model}_N{args.N_dec}'
        else:
            run_name = f'{args.code_type}_{args.code_n}_{args.code_k}_d{args.d_model}'

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'd_model': args.d_model,
                'N_dec': args.N_dec,
                'h': args.h,
                'dropout': args.dropout,
                'dropout_attn': args.dropout_attn,
                'code_type': args.code_type,
                'code_n': args.code_n,
                'code_k': args.code_k,
                'mixture': args.mixture,
                'seed': args.seed,
            },
            tags=['FECCT', 'transfer-analysis', args.mixture or f'{args.code_type}_{args.code_n}_{args.code_k}'],
        )
        logging.info(f'Wandb initialized: {wandb.run.name}')

    # Build code list based on mixture or single code
    if args.mixture:
        code_specs = MIXTURES[args.mixture]
        codes = [Code(ct, n, k) for ct, n, k in code_specs]
        logging.info(f'Training mixture {args.mixture}: {codes}')
    else:
        codes = [Code(args.code_type, args.code_n, args.code_k)]
        logging.info(f'Training single code: {codes[0]}')

    # Create model
    model = FECCT_Decoder(args).to(device)
    logging.info(model)
    logging.info(f'# of Parameters: {sum(p.numel() for p in model.parameters()):,}')

    if args.eval_only:
        # Load checkpoint and evaluate
        model = torch.load(args.checkpoint, map_location=device)
        logging.info(f'Loaded checkpoint: {args.checkpoint}')

        EbNo_range = range(4, 8)
        for code in codes:
            results = evaluate(model, device, code, EbNo_range, use_wandb=use_wandb)
        if use_wandb:
            wandb.finish()
        return

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    EbNo_range_train = range(2, 8)
    EbNo_range_test = range(4, 7)

    # Create training data loader
    if len(codes) == 1:
        # Single code training
        code = codes[0]
        std_train = [EbN0_to_std(eb, code.k / code.n) for eb in EbNo_range_train]
        train_dataset = FECCT_Dataset(code, std_train, length=args.batch_size * 1000, zero_cw=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, collate_fn=collate_single_code
        )
    else:
        # Multi-code mixture training
        # Use average rate for SNR calculation (approximation)
        avg_rate = np.mean([c.k / c.n for c in codes])
        std_train = [EbN0_to_std(eb, avg_rate) for eb in EbNo_range_train]
        train_dataset = MultiCodeDataset(codes, std_train, samples_per_code=args.batch_size * 300, zero_cw=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, collate_fn=collate_single_code
        )

    # Training loop
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train_epoch(
            model, device, train_loader, optimizer, epoch,
            lr=scheduler.get_last_lr()[0], use_wandb=use_wandb
        )
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model.pt'))
            if use_wandb:
                wandb.run.summary['best_loss'] = best_loss
                wandb.run.summary['best_epoch'] = epoch

        # Periodic evaluation
        if epoch % 100 == 0 or epoch == args.epochs:
            logging.info(f'\n=== Evaluation at epoch {epoch} ===')
            for code in codes:
                evaluate(model, device, code, EbNo_range_test, min_errors=50,
                        use_wandb=use_wandb, epoch=epoch)

    # Final save
    torch.save(model, os.path.join(args.path, 'final_model.pt'))
    logging.info(f'\nTraining complete. Models saved to {args.path}')

    # Finish Wandb run
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FECCT Transfer Analysis Training')

    # Training args (defaults match FECCT paper, epochs reduced to 1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='BCH',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=45)
    parser.add_argument('--code_n', type=int, default=63)

    # Mixture args (for behavioral transfer matrix experiments)
    parser.add_argument('--mixture', type=str, default=None,
                        choices=['A', 'B', 'L', 'C', 'D', 'E', 'F'],
                        help='Train on predefined mixture: A=BCH, B=Polar, L=LDPC, C=BCH+Polar, D=BCH+LDPC, E=Polar+LDPC, F=All')

    # Model args (defaults match FECCT paper)
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropout_attn', type=float, default=0.1)

    # Eval args
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Wandb args
    parser.add_argument('--wandb', action='store_true', help='Enable Wandb logging')
    parser.add_argument('--wandb_project', type=str, default='FECCT-Transfer', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity/team name')

    args = parser.parse_args()

    set_seed(args.seed)

    # Create output directory
    if args.mixture:
        exp_name = f'Mix{args.mixture}'
    else:
        exp_name = f'{args.code_type}_N{args.code_n}_K{args.code_k}'

    model_dir = os.path.join(
        'Results_FECCT',
        exp_name + '__' + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir

    # Setup logging
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'training.log')),
        logging.StreamHandler()
    ]
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
    logging.info(f"Output directory: {model_dir}")
    logging.info(f"Args: {args}\n")

    main(args)
