"""C25 (проект): настоящие веса Qwen3-0.6B в своей модели — загрузка и генерация.

Реализуй:
1. load_safetensors(model_dir) — все тензоры чекпойнта (один файл или шарды по index.json);
2. load_pretrained(model_dir, dtype) — config.json + веса → готовая Model в режиме eval;
3. generate(model, idx, max_new_tokens, ...) — генерация с KV-cache и остановкой на eos.

config_from_hf, convert_state_dict, check_state_dict — эталоны C24; sample_next — эталон C25a;
Model — модель из конфига (llmlab/core/model.py).
"""

import json
from pathlib import Path

import torch

from llmlab.core.hf import check_state_dict, config_from_hf, convert_state_dict   # эталоны C24
from llmlab.core.hf import sample_next                                            # эталон C25a
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
