## TL;DR

Llama — тот же Pre-Norm decoder-only трансформер, что GPT-2, но почти с другими деталями:
RMSNorm вместо LayerNorm, RoPE внутри attention вместо `wpe`, GQA вместо MHA, SwiGLU вместо
GELU-MLP, ни одного bias, у Llama 3 8B отдельный `lm_head`. Блок: `x = x + attn(norm₁(x))`,
`x = x + mlp(norm₂(x))` — через норму идёт только вход подслоя, остаточный поток — напрямую.
Модель: `embed_tokens` → слои (каждый со своим KV-cache) → финальная `norm` → `lm_head`.
70 % параметров Llama 3 8B — в SwiGLU, 17 % — в attention, 13 % — в эмбеддингах и голове.

## Формула

$$h = x + \operatorname{Attn}(\operatorname{RMSNorm}_1(x)),\qquad y = h + \operatorname{SwiGLU}(\operatorname{RMSNorm}_2(h))$$

$$\text{logits} = W_{\text{head}}\,\operatorname{RMSNorm}\big(\operatorname{Block}_{L-1} \circ \dots \circ \operatorname{Block}_0(E[\text{idx}])\big)$$

## Формы тензоров

```
idx [B,T] → embed_tokens → [B,T,C] → L × (x + attn(norm(x), cache_ℓ); x + mlp(norm(x))) → [B,T,C]
→ norm → lm_head → [B,T,V]
```

## Что поменялось относительно GPT-2

Скелет тот же (Pre-Norm, финальная норма, остаточный поток). Заменены все детали: позиции —
RoPE в каждом слое вместо `wpe` на входе; LayerNorm → RMSNorm; MHA → GQA с KV-cache; GELU-MLP →
SwiGLU; bias убран везде; выходной слой у Llama 3 8B не связан с эмбеддингами. Имена HF:
`input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp`, `embed_tokens`, `norm`, `lm_head`.
