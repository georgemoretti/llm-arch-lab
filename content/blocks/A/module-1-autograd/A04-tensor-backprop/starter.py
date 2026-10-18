"""Backprop руками на тензорах: Linear, ReLU, softmax + cross-entropy и двухслойный MLP.

Каждый слой — пара функций: forward возвращает выход и cache, backward получает градиент по выходу
и возвращает градиенты по входу и параметрам. torch.autograd (.backward(), torch.autograd.grad) и готовые
softmax / log_softmax / logsumexp / cross_entropy в этой задаче запрещены — тесты это проверяют.
"""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """y = x Wᵀ + b.  x [N, D_in], W [D_out, D_in], b [D_out] → y [N, D_out], cache."""
    # TODO: выход и cache — что из прямого прохода понадобится для dx и dW?
    raise NotImplementedError("linear_forward")


def linear_backward(dy: torch.Tensor, cache):
    """dy [N, D_out] → dx [N, D_in], dW [D_out, D_in], db [D_out]."""
    # TODO 1: dx — какая матрица переводит [N, D_out] обратно в [N, D_in]?
    # TODO 2: dW должен иметь форму W. Собери её из dy и x (подсказка: сверь формы).
    # TODO 3: b прибавлялся к каждой строке батча — что делать с градиентами строк?
    raise NotImplementedError("linear_backward")


def relu_forward(x: torch.Tensor):
    # TODO: выход и cache.
    raise NotImplementedError("relu_forward")


def relu_backward(dy: torch.Tensor, cache) -> torch.Tensor:
    # TODO: градиент проходит там, где вход был > 0, и обнуляется в остальных местах.
    raise NotImplementedError("relu_backward")


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Средний по батчу −log p[target] и его градиент по логитам.

    logits [N, V], targets [N] (индексы классов) → loss [] (скаляр-тензор), dlogits [N, V].
    """
    # TODO 1: log-softmax без переполнения: вычти из каждой строки её максимум (почему ответ не изменится?).
    # TODO 2: loss = −среднее log p[i, targets[i]] (fancy-индексация: log_probs[torch.arange(N), targets]).
    # TODO 3: градиент по логитам — softmax минус one-hot цели, и не забудь, что лосс — СРЕДНЕЕ по N.
    raise NotImplementedError("softmax_cross_entropy")


def mlp_loss_and_grads(x: torch.Tensor, targets: torch.Tensor, params: dict[str, torch.Tensor]):
    """Linear → ReLU → Linear → softmax-CE. params: W1 [H, D], b1 [H], W2 [V, H], b2 [V].

    Возвращает (loss, grads) — grads с теми же ключами и формами, что params.
    """
    # TODO: прямой проход с сохранением cache, затем backward в обратном порядке.
    raise NotImplementedError("mlp_loss_and_grads")
