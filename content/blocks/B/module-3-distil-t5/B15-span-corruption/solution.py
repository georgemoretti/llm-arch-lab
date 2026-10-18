"""Span corruption T5: маска шума из спанов средней длины 3, sentinel-токены во входе и в цели.

Эталон. Алгоритм — как в T5 (random_spans_noise_mask, noise_span_to_unique_sentinel).
"""

from __future__ import annotations

import torch


def random_segmentation(num_items: int, num_segments: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Случайное разбиение num_items на num_segments непустых кусков: длины [num_segments], сумма = num_items."""
    cuts = torch.randperm(num_items - 1, generator=generator)[: num_segments - 1].sort().values + 1
    bounds = torch.cat([torch.tensor([0]), cuts, torch.tensor([num_items])])
    return bounds.diff()


def random_spans_noise_mask(length: int, noise_density: float = 0.15, mean_noise_span_length: float = 3.0,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Маска шума T5 (bool [length]): round(length · density) токенов в round(·/mean) спанах.

    Спаны текста и шума чередуются; первым идёт текст, последним — шум (как в T5).
    """
    num_noise = min(max(int(round(length * noise_density)), 1), length - 1)
    num_spans = max(int(round(num_noise / mean_noise_span_length)), 1)
    noise_lengths = random_segmentation(num_noise, num_spans, generator)
    text_lengths = random_segmentation(length - num_noise, num_spans, generator)
    lengths = torch.stack([text_lengths, noise_lengths], dim=1).reshape(-1)   # текст, шум, текст, шум, …
    flags = torch.tensor([False, True]).repeat(num_spans)
    return flags.repeat_interleave(lengths)


def noise_span_to_unique_sentinel(ids: torch.Tensor, noise_mask: torch.Tensor, sentinels: list[int]) -> torch.Tensor:
    """Каждый спан шума → один sentinel (i-й спан → sentinels[i]); остальные токены — как есть. [L] → [L']."""
    prev_is_noise = torch.cat([noise_mask.new_zeros(1), noise_mask[:-1]])
    first = noise_mask & ~prev_is_noise                 # первый токен спана — станет sentinel
    rest = noise_mask & prev_is_noise                   # остальные токены спана — выкидываем
    span = torch.cumsum(first.long(), 0) - 1            # номер спана у каждого токена
    if int(first.sum()) > len(sentinels):
        raise ValueError(f"спанов {int(first.sum())}, а sentinel-токенов только {len(sentinels)}")
    table = torch.tensor(sentinels, dtype=ids.dtype, device=ids.device)
    out = torch.where(first, table[span.clamp(min=0, max=len(sentinels) - 1)], ids)
    return out[~rest]


def span_corruption(ids: torch.Tensor, noise_mask: torch.Tensor, sentinels: list[int],
                    eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Вход и цель T5: во входе спаны шума заменены sentinel-ами, в цели — наоборот, спаны текста; обе с EOS.

    Выравнивание sentinel-ов держится на том, что маска начинается с текста (так строит random_spans_noise_mask).
    """
    eos = torch.tensor([eos_id], dtype=ids.dtype, device=ids.device)
    inputs = torch.cat([noise_span_to_unique_sentinel(ids, noise_mask, sentinels), eos])
    targets = torch.cat([noise_span_to_unique_sentinel(ids, ~noise_mask, sentinels), eos])
    return inputs, targets
