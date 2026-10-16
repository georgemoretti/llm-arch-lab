## TL;DR

QK-Norm — RMSNorm на q и k перед скалярным произведением (и до RoPE). После нормы длина q и k
фиксирована, логиты ограничены ($|\text{score}| \le \sqrt{D}$ при единичных весах) и не зависят от
масштаба $W_Q, W_K$: рост логитов во время обучения больше не может «сломать» softmax. Gemma 3 и
Qwen3 нормируют каждую голову (вес размера $D$), OLMo 2 — всю проекцию $H \cdot D$ до разбиения на
головы. Gemma 3 заменила этим soft-capping логитов из Gemma 2. Порядок «норма → RoPE» важен: с
обученными весами нормы операции не коммутируют.

## Формула

$$\hat q = w_q \odot \frac{q}{\operatorname{RMS}(q)},\quad \hat k = w_k \odot \frac{k}{\operatorname{RMS}(k)},\quad
\text{score} = \frac{(R\hat q)\cdot(R\hat k)}{\sqrt D},\quad |\text{score}| \le \sqrt{D}\ \text{при } w = 1$$

## Формы тензоров

```
q_proj(x) [B,T,H·D] ─("full": norm)→ view → [B,H,T,D] ─("head": norm)→ RoPE → scores [B,H,T,T]
```

## Что поменялось относительно C04

Две RMSNorm в attention: на q и на k (v не трогаем), до RoPE. Остальное — GQA из C04 с RoPE.
