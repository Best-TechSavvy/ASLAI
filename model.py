# ============================================================
# model.py
# ASL model architecture — must match training exactly
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCNN(nn.Module):
    """Per-frame feature extractor. Input: (B, T, 390) → Output: (B, T, 256)"""
    def __init__(self, input_dim=390, embed_dim=256, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size, padding=pad),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size, padding=pad),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(256, embed_dim, kernel_size, padding=pad),
            nn.BatchNorm1d(embed_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ASLModel(nn.Module):
    """
    CNN + BiLSTM with two output heads:
      fingerspelling : CTC over 60-char vocabulary
      signs          : CrossEntropy over 250 ASL sign classes
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.cnn = SpatialCNN(
            input_dim   = cfg['input_dim'],
            embed_dim   = cfg['embed_dim'],
            kernel_size = cfg['cnn_kernel'],
        )
        self.lstm = nn.LSTM(
            input_size    = cfg['embed_dim'],
            hidden_size   = cfg['lstm_hidden'],
            num_layers    = cfg['lstm_layers'],
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg['lstm_dropout'] if cfg['lstm_layers'] > 1 else 0.0,
        )
        lstm_out = cfg['lstm_hidden'] * 2   # 512 (bidirectional)

        self.fspell_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_out, cfg['vocab_size'])
        )
        self.signs_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_out, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg['num_signs'])
        )

    def forward(self, x, seq_lens=None, mode='fingerspelling'):
        """
        Args:
            x        : (B, T, 130, 3)
            seq_lens : (B,) actual frame counts — None for single-sequence inference
            mode     : 'fingerspelling' | 'signs'
        """
        B, T, L, C = x.shape
        x = self.cnn(x.reshape(B, T, L * C))

        if seq_lens is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        if seq_lens is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(
                x, batch_first=True, total_length=T)

        if mode == 'fingerspelling':
            return F.log_softmax(self.fspell_head(x), dim=-1)  # (B, T, vocab)
        else:
            if seq_lens is not None:
                mask   = torch.arange(T, device=x.device)[None, :] < seq_lens[:, None]
                pooled = (x * mask.unsqueeze(-1).float()).sum(1) \
                       / mask.sum(1, keepdim=True).float().clamp(min=1)
            else:
                pooled = x.mean(1)
            return self.signs_head(pooled)                      # (B, num_signs)


def load_model(weights_path: str, device: str = 'cpu') -> ASLModel:
    """
    Load a trained ASLModel from a weights file.

    Args:
        weights_path : path to asl_model_weights_v2.pt
        device       : 'cpu' or 'cuda'

    Returns:
        model in eval mode, on the specified device
    """
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg  = ckpt['cfg']

    # Fill in defaults for any keys not saved in older checkpoints
    cfg.setdefault('input_dim',    390)
    cfg.setdefault('embed_dim',    256)
    cfg.setdefault('cnn_kernel',   3)
    cfg.setdefault('lstm_hidden',  256)
    cfg.setdefault('lstm_layers',  2)
    cfg.setdefault('lstm_dropout', 0.3)
    cfg.setdefault('vocab_size',   60)
    cfg.setdefault('num_signs',    250)

    model = ASLModel(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model
