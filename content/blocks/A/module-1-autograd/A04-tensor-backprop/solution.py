"""Backprop руками на тензорах: Linear, ReLU, softmax + cross-entropy и двухслойный MLP.

Эталон. Каждый слой — пара функций: forward возвращает выход и cache (то, что понадобится обратному
проходу), backward получает градиент по выходу и возвращает градиенты по входу и параметрам.
"""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """y = x Wᵀ + b.  x [N, D_in], W [D_out, D_in], b [D_out] → y [N, D_out], cache."""
    return x @ W.T + b, (x, W)


def linear_backward(dy: torch.Tensor, cache):
    """dy [N, D_out] → dx [N, D_in], dW [D_out, D_in], db [D_out]."""
    x, W = cache
    dx = dy @ W                 # [N, D_out] @ [D_out, D_in]
    dW = dy.T @ x               # [D_out, N] @ [N, D_in] — сумма по примерам батча
    db = dy.sum(dim=0)          # b добавлялся к каждой строке → градиенты строк складываются
    return dx, dW, db


def relu_forward(x: torch.Tensor):
    return x.clamp(min=0), x


def relu_backward(dy: torch.Tensor, cache) -> torch.Tensor:
    x = cache
    return dy * (x > 0).to(dy.dtype)


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Средний по батчу −log p[target] и его градиент по логитам.

    logits [N, V], targets [N] (индексы классов) → loss [] (скаляр-тензор), dlogits [N, V].
    """
    N = logits.shape[0]
    shifted = logits - logits.max(dim=1, keepdim=True).values         # max → 0: exp не переполнится
    log_z = shifted.exp().sum(dim=1, keepdim=True).log()               # [N, 1]
    log_probs = shifted - log_z                                        # log softmax
    loss = -log_probs[torch.arange(N), targets].mean()
    dlogits = log_probs.exp()                                          # softmax
    dlogits[torch.arange(N), targets] -= 1.0                           # p − onehot(y)
    return loss, dlogits / N                                           # / N: лосс — среднее по батчу


def mlp_loss_and_grads(x: torch.Tensor, targets: torch.Tensor, params: dict[str, torch.Tensor]):
    """Linear → ReLU → Linear → softmax-CE. params: W1 [H, D], b1 [H], W2 [V, H], b2 [V].

    Возвращает (loss, grads) — grads с теми же ключами и формами, что params.
    """
    h, c1 = linear_forward(x, params["W1"], params["b1"])
    a, c2 = relu_forward(h)
    logits, c3 = linear_forward(a, params["W2"], params["b2"])
    loss, dlogits = softmax_cross_entropy(logits, targets)
    da, dW2, db2 = linear_backward(dlogits, c3)
    dh = relu_backward(da, c2)
    _, dW1, db1 = linear_backward(dh, c1)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
