"""C25 (проект): настоящие веса Qwen3-0.6B в своей модели — загрузка, сэмплирование, генерация.

Реализуй:
1. load_safetensors(model_dir) — все тензоры чекпойнта (один файл или шарды по index.json);
2. load_pretrained(model_dir, dtype) — config.json + веса → готовая Model в режиме eval;
3. sample_next(logits, temperature, top_k, top_p, generator) — выбор следующего токена;
4. generate(model, idx, max_new_tokens, ...) — генерация с KV-cache и остановкой на eos.

config_from_hf, convert_state_dict, check_state_dict — эталоны C24; Model — модель из конфига (llmlab/core/model.py).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from llmlab.core.hf import check_state_dict, config_from_hf, convert_state_dict   # эталоны C24
from llmlab.core.model import Model                                              # модель из конфига (Фаза 8)


def load_safetensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Все тензоры из model.safetensors или из шардов по model.safetensors.index.json."""
    # TODO: from safetensors.torch import load_file.
    #   Если есть model.safetensors.index.json — в нём weight_map {имя тензора: имя файла}; прочитай каждый файл
    #   один раз и объедини словари. Иначе — один файл model.safetensors.
    raise NotImplementedError("load_safetensors")


def load_pretrained(model_dir: str | Path, dtype: torch.dtype = torch.float32) -> Model:
    """Папка с config.json и *.safetensors → Model с этими весами (eval-режим)."""
    # TODO 1: config.json → dict → config_from_hf → Model(cfg).
    #         (Необязательно: соберите модель под `with torch.device("meta"):` — без случайной инициализации
    #          0.6 B параметров, тогда загружайте load_state_dict(sd, assign=True).)
    # TODO 2: load_safetensors → convert_state_dict(..., hf["model_type"], dtype) → check_state_dict → load_state_dict.
    # TODO 3: если cfg.tie_embeddings — убедись, что lm_head.weight и embed_tokens.weight остались ОДНИМ тензором.
    # TODO 4: вернуть model.eval().
    raise NotImplementedError("load_pretrained")


def sample_next(logits: torch.Tensor, temperature: float = 0.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] → следующий токен [B, 1]. temperature = 0 — argmax; иначе softmax(logits / T),
    обрезанный top-k и top-p (nucleus: наименьший набор токенов с суммарной вероятностью ≥ p)."""
    # TODO 1: temperature == 0 → argmax (keepdim, чтобы форма была [B, 1]).
    # TODO 2: logits / temperature (во float32).
    # TODO 3: top_k — всё, что меньше k-го по величине логита, → −inf.
    # TODO 4: top_p — отсортируй по убыванию, softmax, накопленная сумма. Токен выбрасывается, если масса токенов
    #         ВЫШЕ него уже ≥ p (самый вероятный остаётся всегда). Верни маску на исходные места словаря.
    # TODO 5: torch.multinomial(softmax(...), 1, generator=generator).
    raise NotImplementedError("sample_next")


@torch.no_grad()
def generate(model: Model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
             top_k: int | None = None, top_p: float | None = None, eos_token_id: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """idx [B, T] → [B, T + n], n ≤ max_new_tokens. С KV-cache. После eos строка дополняется eos;
    генерация останавливается, когда eos выдали все строки."""
    # TODO 1: caches = model.new_caches(); prefill: logits = model(idx, caches).
    # TODO 2: шаг: sample_next по логитам последней позиции → [B, 1]; дописать к idx.
    # TODO 3: eos: строки, которые уже закончили, получают eos; если закончили все — выйти.
    # TODO 4: следующий прогон — только новый токен: model(next_token, caches). После последнего токена прогон не нужен.
    raise NotImplementedError("generate")
