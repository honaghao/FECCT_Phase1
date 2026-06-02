"""
FECCT-style Decoder for Transfer Analysis Paper (Option 5)
Based on: "A Foundation Model for Error Correction Codes" (ICLR 2024)
Adapted from: https://github.com/yoniLc/E2E_DC_ECCT

Key FECCT innovations:
1. Code-invariant embedding (only 5 syndrome vectors + 1 magnitude param)
2. Tanner-graph distance attention modulation via learned psi(d)
3. Parity-check-aware output head
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def sign_to_bin(x):
    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention with FECCT's learned distance-to-attention mapping.

    Key innovation: Instead of a binary mask, uses a learned MLP psi(d) that maps
    Tanner-graph distances to attention biases. This makes the model code-invariant.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # FECCT innovation: learned psi(d) mapping from distance to attention bias
        d_hidden = 50
        self.one_d_mapping = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )

    def get_mask_from_pc_matrix(self, pc_matrix):
        """
        Compute attention modulation from parity-check matrix structure.

        Creates a distance-based attention bias using:
        - H @ H^T: co-check relationships (which bits share checks)
        - H^T @ H: co-variable relationships (which checks share bits)

        The learned one_d_mapping converts these to attention biases.
        """
        mask_nk_nk = pc_matrix @ pc_matrix.T
        mask_n_n = pc_matrix.T @ pc_matrix
        tmp1 = torch.cat([mask_n_n, pc_matrix.T], 1)
        tmp2 = torch.cat([pc_matrix, mask_nk_nk], 1)
        combined = torch.cat([tmp1, tmp2], 0)
        return self.one_d_mapping(combined.unsqueeze(-1)).squeeze().unsqueeze(0).unsqueeze(0)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply learned distance-based attention modulation
        scores = scores * self.get_mask_from_pc_matrix(mask)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class FECCT_Decoder(nn.Module):
    """
    Foundation ECC Transformer Decoder.

    Code-invariant architecture that can generalize to unseen codes.
    Key features:
    - Fixed number of embedding parameters (independent of code length)
    - Learned attention modulation based on Tanner-graph structure
    - Parity-check-aware output head
    """
    def __init__(self, args):
        super(FECCT_Decoder, self).__init__()
        self.args = args
        c = copy.deepcopy

        dropout = getattr(args, 'dropout', 0.0)
        dropout_attn = getattr(args, 'dropout_attn', 0.1)

        attn = MultiHeadedAttention(args.h, args.d_model, dropout=dropout_attn)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)

        # FECCT innovation 1: Code-invariant embeddings
        # Only 5 vectors for syndrome (vs n+m unique positions in ECCT)
        self.src_embed_synd = torch.nn.Embedding(5, args.d_model)
        # Single parameter for magnitude embedding
        self.src_embed_magn = torch.nn.Parameter(torch.empty((1, args.d_model)))

        self.decoder = Encoder(
            EncoderLayer(args.d_model, c(attn), c(ff), dropout),
            args.N_dec
        )

        self.oned_final_embed = torch.nn.Sequential(nn.Linear(args.d_model, 1))

        # FECCT innovation 3: Parity-check-aware output head
        self.synd_to_mag = nn.Linear(args.d_model, args.d_model)
        self.mag_to_mag = nn.Linear(args.d_model, args.d_model)

        for name, p in self.named_parameters():
            if p.dim() > 1 and 'src_embed_synd' not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome, pc_matrix):
        """
        Forward pass for FECCT decoder.

        Args:
            magnitude: |y| channel output magnitudes, shape (batch, n)
            syndrome: s = H @ hard_decision(y) mod 2, shape (batch, n-k)
            pc_matrix: Parity-check matrix H, shape (n-k, n)

        Returns:
            z_pred: Predicted noise pattern logits, shape (batch, n)
        """
        # Embed magnitude (code-invariant: same param for all positions)
        emb_magn = self.src_embed_magn.unsqueeze(0) * magnitude.unsqueeze(-1)

        # Embed syndrome (code-invariant: only 5 possible embeddings)
        emb_synd = self.src_embed_synd(sign_to_bin(syndrome).long())

        # Concatenate magnitude and syndrome embeddings
        emb = torch.cat([emb_magn, emb_synd], 1)

        # Pass through transformer with PC-matrix-based attention
        emb = self.decoder(emb, pc_matrix)

        # Parity-check-aware output head
        n = magnitude.size(1)
        emb_mag = emb[:, :n]
        emb_synd = emb[:, n:]

        # Aggregate syndrome info back to bit positions via PC matrix
        tmp = emb_synd.unsqueeze(-1) * pc_matrix.unsqueeze(0).unsqueeze(2)
        z_pred = self.mag_to_mag(emb_mag) + self.synd_to_mag(tmp.permute(0, 1, 3, 2)).sum(1)
        z_pred = self.oned_final_embed(z_pred).squeeze(-1)

        return z_pred

    def loss(self, z_pred, z_gt, y):
        """
        Compute BCE loss and predicted codeword.

        Args:
            z_pred: Predicted noise logits
            z_gt: Ground truth noise (y * sign(x))
            y: Channel output

        Returns:
            loss: BCE loss
            x_pred: Predicted transmitted codeword
        """
        loss = F.binary_cross_entropy_with_logits(
            z_pred, sign_to_bin(torch.sign(z_gt))
        )
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred


if __name__ == '__main__':
    pass
