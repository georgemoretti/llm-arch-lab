"""Эталон: Rotary Position Embeddings (RoPE), раскладка «половинами», как в HF Llama."""

import torch


def rope_inv_freq(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Частоты пар координат: θ^(−2i/D), i = 0..D/2−1 → [D/2], float32."""
    assert head_dim % 2 == 0, "head_dim должен быть чётным: координаты вращаются парами"
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim   # 2i/D
    return theta ** (-exponents)                                               # от 1 до ≈ 1/θ


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = 10000.0):
    """cos и sin углов m·θ_i для каждой позиции m → два тензора [T, D].

    Раскладка «половинами»: координата j и j + D/2 образуют пару и делят одну частоту,
    поэтому угол повторяется дважды: [углы | углы].
    """
    inv_freq = rope_inv_freq(head_dim, theta)                     # [D/2]
    angles = positions.float()[:, None] * inv_freq[None, :]       # [T, D/2]
    emb = torch.cat((angles, angles), dim=-1)                     # [T, D]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1 | x2] → [−x2 | x1] по последней оси (поворот каждой пары на 90°)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Поворачивает q или k: x [B, H, T, D], cos/sin [T, D] → [B, H, T, D]."""
    # Для пары (a, b) = (x_j, x_{j+D/2}): (a·cos − b·sin, b·cos + a·sin) — поворот на угол m·θ_j.
    return x * cos + rotate_half(x) * sin
