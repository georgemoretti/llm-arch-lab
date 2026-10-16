## TL;DR

Gated attention (Qwen, NeurIPS 2025; стоит в Qwen3-Next) умножает выход attention поэлементно на
$\sigma(W_g x)$ — гейт свой у каждой головы и каждого запроса, его веса — вторая половина `q_proj` каждой
головы. Гейт добавляет нелинейность между $W_V$ и $W_O$ и даёт голове «промолчать», поэтому не нужен attention
sink; авторы сообщают о более стабильном обучении и лучшем длинном контексте. Partial RoPE вращает только
первые $R = D \cdot 0.25$ каналов головы (64 из 256 у Qwen3-Next), остальные сравнивают содержание без позиций.

## Формула

$$y = W_O\big[\operatorname{Attn}_h(x) \odot \sigma(g_h)\big]_{h=1}^{H},\qquad
\operatorname{pRoPE}_m(u) = [\operatorname{RoPE}_m(u_{0:R});\ u_{R:D}]$$

## Формы тензоров

```
q_proj(x) [B,T,H·2D] → view [B,T,H,2D] → q, gate [B,T,H,D];  q,k: norm → pRoPE(:R) → attention [B,H,T,D] → [B,T,H·D] ⊙ σ(gate) → o_proj
```

## Что поменялось относительно C09

`q_proj` вдвое шире (запрос + гейт в каждой голове), выход attention гейтируется сигмоидой до `o_proj`,
RoPE — только на четверти каналов, `head_dim` задаётся явно.
