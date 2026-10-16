## TL;DR

Causal self-attention: каждый токен строит запрос, сравнивает его с ключами всех **предыдущих**
токенов (и себя), превращает сходства в веса через softmax и берёт взвешенное среднее значений.
Multi-head делает это H раз параллельно в подпространствах размера D = C / H, затем головы
склеиваются и проецируются обратно. Это базовый слой GPT-2, от которого блок C отсчитывает все
изменения.

## Формула

$$\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D}} + M\right)V,\quad M_{ts} = -\infty \text{ при } s > t$$

$$\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1,\dots,\text{head}_H)\,W_O$$

## Формы тензоров

```
x [B,T,C] → c_attn → [B,T,3C] → split → q,k,v [B,T,C]
→ view(B,T,H,D).transpose(1,2) → [B,H,T,D]
→ q@kᵀ/√D → scores [B,H,T,T] → mask, softmax(-1) → @v → [B,H,T,D]
→ transpose(1,2).contiguous().view(B,T,C) → c_proj → [B,T,C]
```

## Что поменялось относительно GPT-2

Ничего — это и есть GPT-2. Позиционной информации внутри attention нет (позиции добавляются
эмбеддингами на входе модели); у каждой головы свои K и V; маска — полная causal. Всё это
будет меняться: RoPE (C03), GQA (C04), KV-cache (C05), скользящее окно (C07).
