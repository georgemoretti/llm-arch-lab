## Подсказка 1

Начни с того, чтобы увидеть обе стороны: `sorted(Model(cfg).state_dict())` и `sorted(hf_sd)` (в тесте или на
реальном файле через `safetensors.torch.load_file`). Для Qwen3 / Llama имена отличаются только префиксом
`model.` у всего, кроме `lm_head.weight`: `name.removeprefix("model.")`. В конфиге не забудь
`hf.get("head_dim") or hidden_size // num_attention_heads` и `qk_norm = "head"` только для `qwen3`.

## Подсказка 2

Связанный выход: в нашей модели `state_dict()` содержит и `embed_tokens.weight`, и `lm_head.weight` (это один
тензор). Поэтому при `cfg.tie_embeddings` положи `out["lm_head.weight"] = out["embed_tokens.weight"]`, а
`lm_head.weight` из файла, если он есть, сначала сравни с эмбеддингами (`torch.equal`) — не совпал → `ValueError`.
Ключи `...rotary_emb.inv_freq` — частоты RoPE, их модель считает сама: пропусти.

## Подсказка 3

GPT-2: разбери имя `h.{i}.<модуль>.<weight|bias>` через `name.split(".", 2)` и `rest.rsplit(".", 1)`.
`c_attn`: `(t.t() if kind == "weight" else t).split(C, dim=0)` → три куска для `q_proj`, `k_proj`, `v_proj`.
Остальные Conv1D-веса (`attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) — `.t().contiguous()`, их bias — как есть.
Имена с окончанием `.attn.bias` / `.attn.masked_bias` пропусти. `lm_head.weight = embed_tokens.weight`.
В конце: `{k: v.to(dtype) for k, v in out.items()}`.
