"""линейное внимание (Katharopoulos et al., 2020) — kernel trick, параллельная и рекуррентная формы.

Реализуй:
1. feature_map — положительное отображение φ, чтобы exp(q·k) заменить на φ(q)·φ(k);
2. linear_attention_noncausal — все ключи для каждого запроса, но без матрицы [T, T] (ассоциативность);
3. linear_attention — causal-форма «как attention», с маской [T, T] (так считают при обучении);
4. linear_attention_recurrent — та же causal-форма как RNN: состояние фиксированного размера.

Softmax в этой задаче запрещён.
"""

import torch
import torch.nn.functional as F


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = elu(x) + 1 > 0 — заменяет exp(q·k) скалярным произведением φ(q)·φ(k)."""
    # TODO: одна строка. Почему не ReLU? Что будет со знаменателем, если все φ(k_s)·φ(q) = 0?
    raise NotImplementedError("feature_map")


def linear_attention_noncausal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """q, k [B, H, T, d], v [B, H, T, dv] → [B, H, T, dv]; каждый запрос видит все ключи, без матрицы [T, T]."""
    # TODO 1: φ к q и k.
    # TODO 2: (φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV): сначала φ(K)ᵀV [B, H, d, dv] и Σ_s φ(k_s) [B, H, d].
    # TODO 3: числитель φ(Q)·(φ(K)ᵀV), знаменатель φ(Q)·Σφ(k) [B, H, T, 1] (+ eps).
    raise NotImplementedError("linear_attention_noncausal")


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Causal, параллельная форма (для обучения): q, k [B, H, T, d], v [B, H, T, dv] → [B, H, T, dv]."""
    # TODO 1: веса A = φ(Q)φ(K)ᵀ [B, H, T, T], оставить s ≤ t. (Почему здесь маска — это 0, а не −inf?)
    # TODO 2: числитель A·V; знаменатель — сумма ТЕХ ЖЕ весов по ключам (+ eps).
    raise NotImplementedError("linear_attention")


def linear_attention_recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state=None, eps: float = 1e-6):
    """Causal, рекуррентная форма (для генерации): токен за токеном, память O(d·dv) вместо O(T).

    state = (S [B, H, d, dv], z [B, H, d]) или None → (out [B, H, T, dv], новый state).
    """
    # TODO 1: φ к q и k; если state нет — S и z нулями нужных форм, иначе продолжить с переданных.
    # TODO 2: цикл по t: S += φ(k_t) v_tᵀ (внешнее произведение), z += φ(k_t).
    # TODO 3: out_t = φ(q_t)ᵀ S / (φ(q_t)ᵀ z + eps). Обновлять до или после чтения?
    # TODO 4: склеить выходы по оси времени (dim=2) и вернуть (out, (S, z)).
    raise NotImplementedError("linear_attention_recurrent")
