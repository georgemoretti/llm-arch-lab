"""Rotary Position Embeddings (RoPE), раскладка «половинами», как в HF Llama.

Реализуй четыре маленькие функции:
1. rope_inv_freq — частоты пар координат;
2. rope_cos_sin — cos и sin углов для заданных позиций;
3. rotate_half — поворот каждой пары на 90°;
4. apply_rope — поворот q или k на угол, зависящий от позиции.

Пара координат — (j, j + D/2): первая половина вектора и вторая половина.
"""

import torch


def rope_inv_freq(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Частоты пар координат: θ^(−2i/D), i = 0..D/2−1 → [D/2], float32."""
    # TODO: сколько всего пар? Показатель степени для пары i — 2i/D (подсказка: arange с шагом 2).
    raise NotImplementedError("rope_inv_freq")


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = 10000.0):
    """cos и sin углов m·θ_i для каждой позиции m → два тензора [T, D]."""
    # TODO 1: углы — каждая позиция [T] × каждая частота [D/2] → какая форма?
    # TODO 2: у координат j и j + D/2 один и тот же угол. Как получить [T, D] из [T, D/2]?
    # TODO 3: верни (cos, sin).
    raise NotImplementedError("rope_cos_sin")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1 | x2] → [−x2 | x1] по последней оси."""
    # TODO: раздели последнюю ось пополам и склей обратно в нужном порядке и с нужным знаком.
    raise NotImplementedError("rotate_half")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Поворачивает q или k: x [B, H, T, D], cos/sin [T, D] → [B, H, T, D]."""
    # TODO: распиши поворот пары (a, b) на угол φ: (a·cos φ − b·sin φ, b·cos φ + a·sin φ).
    #       Какая комбинация x, rotate_half(x), cos и sin даёт это сразу для всех пар?
    raise NotImplementedError("apply_rope")
