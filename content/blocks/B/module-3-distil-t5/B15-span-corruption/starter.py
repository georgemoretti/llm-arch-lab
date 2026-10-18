"""Span corruption T5: маска шума из спанов средней длины 3, sentinel-токены во входе и в цели.

Реализуй четыре функции: случайное разбиение, маску шума, замену спанов на sentinel и сборку пары (вход, цель).
"""

from __future__ import annotations

import torch


def random_segmentation(num_items: int, num_segments: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Случайное разбиение num_items на num_segments непустых кусков: длины [num_segments], сумма = num_items."""
    # TODO: между num_items элементами num_items − 1 «щелей». Выбери из них num_segments − 1 разрезов без повторов
    #       (torch.randperm(num_items − 1, generator=generator)[: num_segments − 1]), отсортируй, +1 — это начала
    #       кусков. Длины — разности соседних границ [0, разрезы…, num_items] (torch.diff).
    raise NotImplementedError("random_segmentation")


def random_spans_noise_mask(length: int, noise_density: float = 0.15, mean_noise_span_length: float = 3.0,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Маска шума T5 (bool [length]): round(length · density) токенов в round(·/mean) спанах.

    Спаны текста и шума чередуются; первым идёт текст, последним — шум (как в T5).
    """
    # TODO 1: num_noise = round(length · noise_density), но не меньше 1 и не больше length − 1.
    # TODO 2: num_spans = round(num_noise / mean_noise_span_length), не меньше 1.
    # TODO 3: длины спанов шума — random_segmentation(num_noise, num_spans), спанов текста —
    #         random_segmentation(length − num_noise, num_spans) (в этом порядке вызовов: так тесты воспроизводимы).
    # TODO 4: чередуй: текст₀, шум₀, текст₁, шум₁, … (torch.stack по оси 1 + reshape) и разверни в маску
    #         (repeat_interleave флагов [False, True, False, True, …] по этим длинам).
    raise NotImplementedError("random_spans_noise_mask")


def noise_span_to_unique_sentinel(ids: torch.Tensor, noise_mask: torch.Tensor, sentinels: list[int]) -> torch.Tensor:
    """Каждый спан шума → один sentinel (i-й спан → sentinels[i]); остальные токены — как есть. [L] → [L']."""
    # TODO 1: prev_is_noise — маска, сдвинутая вправо на 1 (на позиции 0 — False: до начала текста шума нет).
    # TODO 2: first = noise & ~prev (первый токен спана), rest = noise & prev (остальные токены спана).
    # TODO 3: номер спана у каждой позиции — cumsum(first) − 1. Если спанов больше, чем sentinel-ов, — ValueError.
    # TODO 4: на месте first поставь sentinels[номер], остальные id не трогай; выкинь позиции rest.
    raise NotImplementedError("noise_span_to_unique_sentinel")


def span_corruption(ids: torch.Tensor, noise_mask: torch.Tensor, sentinels: list[int],
                    eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Вход и цель T5: во входе спаны шума заменены sentinel-ами, в цели — наоборот, спаны текста; обе с EOS.

    Выравнивание sentinel-ов держится на том, что маска начинается с текста (так строит random_spans_noise_mask).
    """
    # TODO: inputs = noise_span_to_unique_sentinel(ids, noise_mask, …) + [eos];
    #       targets = noise_span_to_unique_sentinel(ids, ~noise_mask, …) + [eos] — та же функция с инвертированной
    #       маской: в цели «шумом» становится сохранённый текст.
    raise NotImplementedError("span_corruption")
