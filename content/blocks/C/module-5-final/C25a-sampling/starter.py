"""C25a: выбор следующего токена — жадно, temperature, top-k, top-p (nucleus).

Реализуй sample_next(logits, temperature, top_k, top_p, generator). Её использует generate в C25.
"""

import torch
import torch.nn.functional as F


def sample_next(logits: torch.Tensor, temperature: float = 0.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] → следующий токен [B, 1]. temperature = 0 — argmax; иначе softmax(logits / T),
    обрезанный top-k и top-p (nucleus: наименьший набор токенов с суммарной вероятностью ≥ p)."""
    # TODO 1: temperature == 0 → argmax (keepdim, чтобы форма была [B, 1]).
    # TODO 2: logits / temperature (во float32).
    # TODO 3: top_k — всё, что меньше k-го по величине логита, → −inf (k не больше размера словаря).
    # TODO 4: top_p — отсортируй по убыванию, softmax, накопленная сумма. Токен выбрасывается, если масса токенов
    #         ВЫШЕ него уже ≥ p (самый вероятный остаётся всегда). Верни маску на исходные места словаря.
    # TODO 5: torch.multinomial(softmax(...), 1, generator=generator).
    raise NotImplementedError("sample_next")
