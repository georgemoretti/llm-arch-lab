# Конфигурируемая модель (`llmlab/core/model.py`)

Одна модель-декодер, которая собирается из YAML. Каждая деталь — норма, позиции, attention, FFN/MoE,
расписание слоёв — выбирается конфигом и берётся из эталонов задач блока C (`llmlab/core/*.py`).
Это база финального проекта (C24, C25) и способ быстро собрать «свою» архитектуру.

```python
from llmlab.core.model import Model, ModelConfig, generate

model = Model.from_config("gemma3-like")              # llmlab/configs/gemma3-like.yaml
cfg = ModelConfig.from_yaml("my-model.yaml")          # свой файл
logits = Model(cfg)(idx)                              # idx [B, T] → [B, T, V]
out = generate(model, idx, max_new_tokens=20)         # жадно, с KV-cache
```

```bash
llmlab model                 # все готовые конфиги: слоёв, d_model, attention, ffn, параметров
llmlab model gemma3-like     # расписание слоёв: mixer, окно, θ RoPE / NoPE, dense / moe
```

## Готовые конфиги (`llmlab/configs/`)

| конфиг | что показывает | откуда числа |
|---|---|---|
| `gpt2` | LayerNorm, learned-позиции, MHA с bias, GELU-MLP | `openai-community/gpt2` config.json |
| `llama-like` | RMSNorm, RoPE, GQA, SwiGLU (игрушечный размер) | пропорции Llama 3 |
| `gemma3-like` | sandwich-нормы (1 + w), QK-Norm по голове, 5 sliding : 1 full, свой θ у sliding-слоёв, GeGLU, √d·эмбеддинги | Gemma 3 1B config.json |
| `olmo2-like` | нормы после подслоя (OLMo 2), QK-Norm по проекции | OLMo 2 config.json |
| `smollm3-like` | NoPE в каждом 4-м слое | SmolLM3 config.json |
| `deepseek-tiny` | MLA + DeepSeekMoE (sigmoid-роутер, shared expert, первые слои dense) | DeepSeek-V3 config.json |
| `hybrid-tiny` | Gated DeltaNet : gated attention = 3 : 1, partial RoPE | Qwen3-Next config.json |
| `qwen3-0.6b` | настоящий размер для C25 | `Qwen/Qwen3-0.6B` config.json (сверено 2026-10-17) |
| `smollm2-135m` | настоящий размер, бонус к C25 (`model_type: llama`) | `HuggingFaceTB/SmolLM2-135M` config.json |

Источник каждого числа — в комментарии в начале файла и в поле `source`. Игрушечные конфиги (`*-like`, `*-tiny`)
берут из реальной модели **устройство** (типы компонентов, соотношения, θ, ε), а не размеры.

## Схема YAML

Неизвестный ключ — ошибка (`extra = "forbid"`): опечатка не превращается в тихий дефолт.

```yaml
name: my-model
source: https://…/config.json        # откуда взяты числа (необязательно)
vocab_size: 256
d_model: 64
n_layers: 6
tie_embeddings: true                 # lm_head.weight = embed_tokens.weight
embed_scale: false                   # Gemma: эмбеддинги × √d_model

norm:
  type: rmsnorm                      # layernorm | rmsnorm | gemma_rmsnorm (вес 1 + w)        C01, C08
  eps: 1.0e-6
  placement: pre                     # pre | post | olmo | sandwich                          C08

position:
  type: rope                         # learned | rope | none                                C03, C10
  theta: 10000
  local_theta: null                  # Gemma 3: θ sliding-слоёв (global берут theta)          C07
  partial_rotary_factor: 1.0         # Qwen3-Next: 0.25 — RoPE на четверти головы           C19
  no_rope_layer_interval: null       # SmolLM3: 4 — каждый 4-й слой без RoPE                 C10
  max_positions: 2048                # только для learned

attention:
  type: gqa                          # gqa (MHA / GQA / MQA) | mla                         C04, C16b
  n_heads: 4
  n_kv_heads: 2                      # null → n_heads (MHA); 1 → MQA
  head_dim: null                     # null → d_model / n_heads (у Qwen3 явно 128)
  bias: false                        # GPT-2: true
  qk_norm: null                      # head (Qwen3, Gemma 3) | full (OLMo 2)                 C09
  output_gate: false                 # Qwen3-Next: σ(gate) ⊙ attn                          C19
  query_scale: null                  # null → 1/√head_dim
  window: null                       # размер окна sliding-слоёв                             C07, C11
  sliding_window_pattern: null       # 6 → 5 sliding : 1 full; null при заданном window → все sliding
  # только для mla: kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim            C16b

linear_attention:                    # необязательно: гибрид                                C21, C22
  type: gated_deltanet
  full_attention_interval: 4         # каждый 4-й слой — attention, остальные — DeltaNet
  n_heads: 4
  head_k_dim: 16
  head_v_dim: 16

ffn:
  type: swiglu                       # gelu (GPT-2, d_hidden = 4·d_model) | swiglu | geglu  C02
  d_hidden: 172
  bias: false

moe:                                 # необязательно: заменяет FFN в части слоёв            C12–C15
  n_routed: 8
  n_shared: 1
  top_k: 2
  d_expert: 32
  scoring_func: softmax              # softmax | sigmoid (DeepSeek-V3)
  norm_topk_prob: true
  routed_scaling_factor: 1.0
  first_k_dense_replace: 1           # первые k слоёв — dense FFN
  moe_layer_freq: 1
```

Проверки при загрузке: `n_heads` делится на `n_kv_heads`; без `head_dim` `d_model` делится на `n_heads`; для `mla`
заданы все четыре размера; `no_rope_layer_interval` — только при `rope`.

## Как модель собирается

1. `layer_specs(cfg)` превращает конфиг в список `LayerSpec(mixer, window, use_rope, rope_theta, ffn)` — по одному на
   слой — через функции-расписания эталонов: `layer_types` (C07), `rope_layers` (C10), `mlp_layer_types` (C15),
   `hybrid_layer_types` (C22). `llmlab model <имя>` печатает именно этот список.
2. `Block(cfg, spec)` берёт mixer, FFN и нормы **из реестра** и соединяет их по схеме `norm.placement`.
3. `Model` = `embed_tokens` (+ `embed_positions`) → блоки → `norm` → `lm_head`.

Имена параметров — как в `transformers` без префикса `model.` (`layers.3.self_attn.q_proj.weight`), поэтому загрузка
весов HF (`llmlab/core/hf.py`, задачи C24/C25) — в основном снятие префикса.

Кеш при генерации — у каждого mixer-а свой, `model.new_caches()` собирает их по слоям: `KVCache` (C05),
`RingKVCache` для sliding-слоёв (C11), `LatentCache` для MLA (C16), состояние фиксированного размера для DeltaNet (C21).

## Реестр компонентов

```python
REGISTRY = {"norm": {...}, "mixer": {...}, "ffn": {...}}
```

| вид | имя | фабрика получает | что строит |
|---|---|---|---|
| `norm` | `layernorm`, `rmsnorm`, `gemma_rmsnorm` | `(dim, eps)` | `nn.Module` нормы |
| `mixer` | `gqa`, `mla`, `gated_deltanet` | `(cfg, spec)` | модуль с `forward(x, cache)` и `new_cache()` |
| `ffn` | `gelu`, `swiglu`, `geglu`, `moe` | `(cfg)` | модуль `[B, T, C] → [B, T, C]` |

### Добавить компонент

Пример: FFN с ReLU² (как в Primer).

```python
# llmlab/core/model.py, рядом с остальными FFN
class ReLU2FFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)) ** 2)


register("ffn", "relu2")(lambda cfg: ReLU2FFN(cfg.d_model, cfg.ffn.d_hidden))
```

Затем:

1. Добавь `"relu2"` в `Literal` поля `FFNConfig.type` — иначе YAML с `type: relu2` не пройдёт валидацию.
2. Если компоненту нужны новые параметры — добавь поле в соответствующую секцию конфига (`FFNConfig`,
   `AttentionConfig`, …) с дефолтом, чтобы старые YAML продолжали работать.
3. Для mixer-а реализуй ещё `new_cache()` и `forward(x, cache)`: `x` — только **новые** токены, кеш хранит прошлое и
   знает `seq_len` (по нему считаются позиции).
4. Если компонент меняет расписание слоёв (как окно или NoPE) — добавь поле в `LayerSpec` и его вычисление в
   `layer_specs`.
5. Тест в `tests/test_core_model.py`: формы, causal (выход не зависит от будущих токенов), генерация с кешем == без
   кеша. Если компонент — частный случай эталона задачи, сверь их на одних весах (так сделано для `Attention`).
6. Если хочешь грузить веса HF с этим компонентом — поддержи его в `config_from_hf` / `convert_state_dict`
   (`llmlab/core/hf.py`) **и** в `solution.py` задачи C24 (`tests/test_core_hf.py` проверяет, что они совпадают).

Почему `Attention` в `model.py` своя, а не один из классов эталонов: каждый эталон показывает одну идею, а реальной
модели нужно их сочетание (Qwen3 = GQA + QK-Norm + `head_dim ≠ d/H`; Gemma 3 = то же + окно + свой θ). Поэтому она
собрана из функций эталонов (`rope_cos_sin`, `apply_partial_rope`, `repeat_kv`, `KVCache`, `RingKVCache`), а
`tests/test_core_model.py` проверяет, что её частные случаи совпадают с `LlamaAttention`, `QKNormAttention`,
SWA, NoPE и `GatedAttention` на одинаковых весах.

## Реальные веса

```bash
uv sync --extra weights                                   # huggingface_hub, tokenizers, transformers
llmlab generate qwen3-0.6b "The capital of France is"     # скачает ~1.5 ГБ в ~/.cache/huggingface
llmlab generate qwen3-0.6b "Name three planets of the solar system" --chat -n 60
llmlab generate smollm2-135m "Once upon a time" -t 0.8 --top-p 0.9
llmlab generate qwen3-0.6b "Hello" --impl workspace      # твоё решение C25 вместо эталона
```

Поддерживаемые `model_type`: `qwen3`, `llama` (Llama 2/3, SmolLM2), `gpt2`. Остальное `config_from_hf` отклоняет с
объяснением, какой опции нет (например, `rope_scaling`).
