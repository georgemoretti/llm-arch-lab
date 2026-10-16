"""C24: config.json и веса чекпойнта Hugging Face → конфиг и state_dict своей модели.

Реализуй:
1. config_from_hf(hf) — dict из config.json → ModelConfig (llmlab/core/model.py) для model_type
   qwen3, llama и gpt2; опции, которых в курсе нет, — ValueError;
2. convert_state_dict(hf_sd, cfg, model_type, dtype) — тензоры чекпойнта → state_dict для Model(cfg).

Как выглядит ModelConfig — смотри готовые конфиги llmlab/configs/qwen3-0.6b.yaml, smollm2-135m.yaml, gpt2.yaml:
config_from_hf должна вернуть ровно их (кроме name / source). check_state_dict ниже — готовый помощник.
"""

import torch

from llmlab.core.model import ModelConfig   # конфиг модели из YAML (Фаза 8)


def config_from_hf(hf: dict) -> ModelConfig:
    """config.json модели Hugging Face (dict) → ModelConfig. Неподдерживаемые опции → ValueError."""
    # TODO 1 (gpt2): поля n_embd, n_layer, n_head, n_positions, layer_norm_epsilon, vocab_size.
    #   norm: layernorm / pre; position: learned (max_positions = n_positions); attention: gqa с bias;
    #   ffn: gelu, d_hidden = 4 · n_embd, bias. Выход GPT-2 всегда связан с wte.
    # TODO 2 (qwen3, llama): hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
    #   head_dim (если есть — у Qwen3 он НЕ равен hidden_size / heads!), intermediate_size, rms_norm_eps,
    #   rope_theta, tie_word_embeddings. У Qwen3 — QK-Norm по голове.
    #   Внимание: transformers ≥ 5 пишет θ в "rope_parameters": {"rope_theta": …, "rope_type": …} — поддержи оба.
    # TODO 3: ValueError для rope_scaling ≠ None (или rope_parameters.rope_type ≠ "default"), use_sliding_window, attention_bias / mlp_bias,
    #   hidden_act ≠ silu, других model_type. Тихо проигнорировать такую опцию — значит получить модель,
    #   которая грузится без ошибок и выдаёт мусор.
    # Подсказка: ModelConfig.model_validate({...вложенные dict как в YAML...}).
    raise NotImplementedError("config_from_hf")


def convert_state_dict(hf_sd: dict[str, torch.Tensor], cfg: ModelConfig, model_type: str,
                       dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Тензоры чекпойнта HF → state_dict ``Model(cfg)``: имена, Conv1D → Linear, связанный lm_head, dtype."""
    # Сравни имена: Model(cfg).state_dict().keys() и ключи hf_sd (в тестах и в реальном файле).
    # TODO 1 (qwen3, llama): снять префикс "model.", выбросить служебные "...rotary_emb.inv_freq".
    #   tie_embeddings: lm_head.weight = embed_tokens.weight; если lm_head.weight в файле есть и отличается —
    #   ValueError. Без связи — lm_head.weight берётся из файла как есть.
    # TODO 2 (gpt2): wte → embed_tokens, wpe → embed_positions, ln_f → norm, h.{i}.ln_1 / ln_2 → нормы блока,
    #   attn.c_attn → q_proj / k_proj / v_proj, attn.c_proj → self_attn.o_proj, mlp.c_fc / mlp.c_proj — те же.
    #   Веса Conv1D хранятся как [in, out] — транспонируй. Буферы h.{i}.attn.bias — это маска, не вес.
    # TODO 3: привести все тензоры к dtype.
    raise NotImplementedError("convert_state_dict")


def check_state_dict(model: torch.nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """Готово: понятная ошибка вместо длинного отчёта load_state_dict — чего не хватает, что лишнее, где не те формы."""
    expected = model.state_dict()
    missing = sorted(set(expected) - set(sd))
    extra = sorted(set(sd) - set(expected))
    shapes = [f"{k}: в чекпойнте {tuple(sd[k].shape)}, в модели {tuple(expected[k].shape)}"
              for k in expected if k in sd and sd[k].shape != expected[k].shape]
    problems = []
    if missing:
        problems.append(f"нет {len(missing)} тензоров, например {missing[:3]}")
    if extra:
        problems.append(f"лишние {len(extra)} тензоров, например {extra[:3]}")
    if shapes:
        problems.append("не совпадают формы: " + "; ".join(shapes[:3]))
    if problems:
        raise ValueError("state_dict не подходит к модели: " + " | ".join(problems))
