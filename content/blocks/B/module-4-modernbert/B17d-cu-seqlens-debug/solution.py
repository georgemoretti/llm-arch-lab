"""Unpadding: батч без PAD — упаковка, cu_seqlens, блочно-диагональная маска, позиции, экономия вычислений.

Эталон.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def unpad_input(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """attention_mask [B, T] (1 / 0) → (indices [N] — номера настоящих токенов в x.view(B·T, …),
    cu_seqlens [B + 1] int32 — границы текстов в упаковке, max_seqlen)."""
    seqlens = attention_mask.sum(-1, dtype=torch.int32)
    indices = attention_mask.flatten().nonzero().flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, int(seqlens.max())


def pack(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """[B, T, …] → [N, …]: только настоящие токены, текст за текстом."""
    return x.flatten(0, 1)[indices]


def unpack(packed: torch.Tensor, indices: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """[N, …] → [B, T, …]; на месте PAD — нули."""
    out = packed.new_zeros(B * T, *packed.shape[1:])
    out[indices] = packed
    return out.view(B, T, *packed.shape[1:])


def packed_position_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Позиции в упаковке [N]: у каждого текста свои 0, 1, 2, … (для RoPE)."""
    N = int(cu_seqlens[-1])
    seq_id = torch.searchsorted(cu_seqlens[1:], torch.arange(N, device=cu_seqlens.device), right=True)
    return torch.arange(N, device=cu_seqlens.device) - cu_seqlens[seq_id].long()


def varlen_mask(cu_seqlens: torch.Tensor, window: int | None = None) -> torch.Tensor:
    """[N, N] bool, блочно-диагональная: токен видит только токены своего текста (и, если задано, в окне ±window // 2)."""
    N = int(cu_seqlens[-1])
    pos = torch.arange(N, device=cu_seqlens.device)
    seq_id = torch.searchsorted(cu_seqlens[1:], pos, right=True)
    mask = seq_id[:, None] == seq_id[None, :]
    if window is not None:
        mask = mask & ((pos[:, None] - pos[None, :]).abs() <= window // 2)
    return mask


def varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                     window: int | None = None) -> torch.Tensor:
    """q, k, v [N, H, D] упакованного батча → [N, H, D]; тексты не видят друг друга."""
    D = q.shape[-1]
    scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1) / D ** 0.5          # [H, N, N]
    scores = scores.masked_fill(~varlen_mask(cu_seqlens, window), torch.finfo(scores.dtype).min)
    return (scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1)


def padding_savings(attention_mask: torch.Tensor) -> dict[str, float]:
    """Какую долю работы на паддинг экономит упаковка: линейные слои ∝ числу токенов, attention ∝ Σ L²."""
    lengths = attention_mask.sum(-1).double()
    B, T = attention_mask.shape
    return {"linear": 1 - float(lengths.sum()) / (B * T),
            "attention": 1 - float((lengths ** 2).sum()) / (B * T * T)}
