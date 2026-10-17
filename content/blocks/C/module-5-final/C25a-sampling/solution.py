"""Эталон C25a: выбор следующего токена — жадно, temperature, top-k, top-p (nucleus)."""

import torch
import torch.nn.functional as F


def sample_next(logits: torch.Tensor, temperature: float = 0.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] → следующий токен [B, 1]. temperature = 0 — argmax; иначе softmax(logits / T),
    обрезанный top-k и top-p (nucleus: наименьший набор токенов с суммарной вероятностью ≥ p)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        before = probs.cumsum(dim=-1) - probs                       # масса токенов СТРОГО выше текущего
        sorted_logits = sorted_logits.masked_fill(before >= top_p, float("-inf"))   # первый токен остаётся всегда
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)
