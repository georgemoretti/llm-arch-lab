# Собрать блок и модель Llama

## Зачем это нужно

Все детали уже есть: RMSNorm (C01), SwiGLU (C02), RoPE (C03), GQA (C04), KV-cache (C05). Осталось
собрать из них блок и модель — и убедиться, что сборка правильная, ведь именно здесь чаще всего
ошибаются при переносе весов: перепутанный порядок норм или остатков не роняет код, но даёт
другую модель.

Сравни с `llmlab/core/gpt2.py`: «скелет» у Llama тот же, что у GPT-2 (Pre-Norm decoder-only
трансформер), а меняются почти все детали.

## Что поменялось относительно GPT-2

| | GPT-2 | Llama 3 |
|---|---|---|
| вход | `wte[tok] + wpe[pos]` | только `embed_tokens[tok]` — позиции в RoPE внутри attention |
| нормализация | LayerNorm (с bias) | RMSNorm (C01), eps = 1e-5 |
| место норм | Pre-Norm + финальная норма | так же: Pre-Norm + финальная `norm` |
| attention | MHA, `c_attn` с bias | GQA + RoPE + KV-cache (C03–C05), без bias |
| FFN | Linear → GELU → Linear, ×4 | SwiGLU (C02), без bias |
| выходной слой | связан с `wte` | Llama 3 8B — **отдельный** `lm_head` (у Llama 3.2 1B — снова связан) |
| имена (HF) | `ln_1`, `attn`, `ln_2`, `mlp` | `input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp` |

Источник: config.json `meta-llama/Meta-Llama-3-8B` (зеркало `NousResearch/Meta-Llama-3-8B`):
`tie_word_embeddings = false`, `rms_norm_eps = 1e-5`, `attention_bias = false`; config.json
`meta-llama/Llama-3.2-1B` (зеркало `unsloth/Llama-3.2-1B`): `tie_word_embeddings = true`.

**Куда уходят 8 млрд параметров Llama 3 8B** (по тем же config.json: d = 4096, 32 слоя,
32/8 голов, FFN 14 336, словарь 128 256):

| часть | на слой | всего | доля |
|---|---|---|---|
| attention: q, o — d×d; k, v — d×1024 | 41,9 млн | 1,34 млрд | 17 % |
| SwiGLU: 3 × d × 14 336 | 176,2 млн | 5,64 млрд | 70 % |
| `embed_tokens` + `lm_head` | — | 1,05 млрд | 13 % |
| нормы | 8 192 | ≈ 0,3 млн | ≈ 0 % |
| **итого** | | **8 030 261 248** | |

## Формула

Блок (Pre-Norm, два остаточных соединения):

$$
h = x + \operatorname{Attn}\big(\operatorname{RMSNorm}_1(x)\big), \qquad
y = h + \operatorname{SwiGLU}\big(\operatorname{RMSNorm}_2(h)\big)
$$

Модель:

$$
x_0 = E[\text{idx}], \quad x_{\ell+1} = \operatorname{Block}_\ell(x_\ell), \quad
\text{logits} = W_{\text{head}}\, \operatorname{RMSNorm}(x_L)
$$

Важно: через нормы проходит только **вход подслоя**, а сам остаточный поток $x$ идёт по модели
«напрямую». Поэтому, если подслой выдаёт ноль, блок — тождественное отображение; это свойство
делает глубокие Pre-Norm сети устойчивыми при обучении.

## Формы тензоров

```shapes
idx [B, T] — только новые токены, если есть кеш
> embed_tokens
x [B, T, C]
> каждый слой ℓ = 1…L со своим caches[ℓ]:
> x = x + self_attn(input_layernorm(x), caches[ℓ]) — [B, T, C]
> x = x + mlp(post_attention_layernorm(x)) — [B, T, C]
x [B, T, C]
> norm → lm_head
logits [B, T, V]
```

:::viz norm-placement:::

## Задание

В `solution.py` (рабочая копия в `workspace/C06-llama-block/`) реализуй:

1. `LlamaBlock.__init__(cfg)` — подмодули `input_layernorm` (`RMSNorm`), `self_attn`
   (`LlamaAttention`), `post_attention_layernorm` (`RMSNorm`), `mlp` (`SwiGLUFFN`);
2. `LlamaBlock.forward(x, cache=None)` — Pre-Norm, как в формуле;
3. `LlamaModel.__init__(cfg)` — `embed_tokens` (`nn.Embedding`), `layers` (`nn.ModuleList`),
   `norm`, `lm_head` (без bias); при `cfg.tie_embeddings` — общий тензор весов;
4. `LlamaModel.forward(idx, caches=None)` — `caches` — список из `n_layers` кешей или `None`.

Имена — как в Hugging Face: тесты загружают в твою модель веса эталона, а в Фазе 8 так же будут
загружаться настоящие веса. Детали импортируются из `llmlab.core.llama` (эталоны C01–C05).
`LlamaConfig`, `new_caches` и `generate` даны.

## Критерии готовности

- имена и формы параметров совпадают с эталоном; нет позиционных эмбеддингов и bias;
- блок совпадает с эталоном; с нулевыми `o_proj` и `down_proj` блок возвращает вход как есть;
- при `tie_embeddings=True` `lm_head` и `embed_tokens` делят один тензор;
- логиты модели совпадают с эталоном, будущее не протекает;
- prefill + по одному токену с кешами == полный прогон, у каждого слоя свой кеш;
- `generate` с кешем и без даёт одинаковые токены.

Запуск: `llmlab test C06-llama-block`.
