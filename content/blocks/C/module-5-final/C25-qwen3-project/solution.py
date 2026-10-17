"""Эталон: настоящие веса Qwen3-0.6B в своей модели — загрузка safetensors и генерация."""

import json
from pathlib import Path

import torch

from llmlab.core.hf import check_state_dict, config_from_hf, convert_state_dict   # эталоны «Чекпойнт Hugging Face»
from llmlab.core.hf import sample_next                                            # эталон «Сэмплирование»
from llmlab.core.model import Model                                              # модель из конфига (Фаза 8)


def load_safetensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Все тензоры из model.safetensors или из шардов по model.safetensors.index.json."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd.update(load_file(model_dir / shard))
    return sd


def load_pretrained(model_dir: str | Path, dtype: torch.dtype = torch.float32) -> Model:
    """Папка с config.json и *.safetensors → Model с этими весами (eval-режим)."""
    model_dir = Path(model_dir)
    hf = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = config_from_hf(hf)
    with torch.device("meta"):                                      # не тратим время на случайную инициализацию
        model = Model(cfg)
    sd = convert_state_dict(load_safetensors(model_dir), cfg, hf["model_type"], dtype)
    check_state_dict(model, sd)
    model.load_state_dict(sd, assign=True)
    if cfg.tie_embeddings:
        model.lm_head.weight = model.embed_tokens.weight            # assign=True разрывает связь — восстанавливаем
    return model.eval()



@torch.no_grad()
def generate(model: Model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
             top_k: int | None = None, top_p: float | None = None, eos_token_id: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """idx [B, T] → [B, T + n], n ≤ max_new_tokens. С KV-cache. После eos строка дополняется eos;
    генерация останавливается, когда eos выдали все строки."""
    caches = model.new_caches()
    logits = model(idx, caches)
    done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    for step in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], temperature, top_k, top_p, generator)        # [B, 1]
        if eos_token_id is not None:
            nxt = nxt.masked_fill(done[:, None], eos_token_id)                    # закончившие строки — eos
            done = done | (nxt[:, 0] == eos_token_id)
        idx = torch.cat((idx, nxt), dim=1)
        if bool(done.all()) or step + 1 == max_new_tokens:
            break                                                                 # лишний прогон не нужен
        logits = model(nxt, caches)                                               # с кешем — только новый токен
    return idx
