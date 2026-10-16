## TL;DR

DSA (DeepSeek-V3.2) делает attention разреженным по содержанию: дешёвый lightning indexer оценивает все пары
(запрос, ключ) как $\sum_h w_{t,h}\operatorname{ReLU}(q_{t,h}\cdot k_s)$, и каждый запрос считает полное attention
только по top-$k$ ключам ($k = 2048$ у DeepSeek-V3.2). Основное attention — $O(T\cdot k)$ вместо $O(T^2)$;
индексатор квадратичен, но лёгкий. KV-cache не уменьшается: выбирать можно только из того, что хранится.
Через top-k градиент не идёт — индексатор учат отдельно, подгоняя под распределение плотного attention.

## Формула

$$I_{t,s} = \sum_h w_{t,h}\operatorname{ReLU}(q^I_{t,h}\cdot k^I_s),\qquad
o_t = \sum_{s\in\operatorname{top-k}(I_{t,\cdot})} \operatorname{softmax}\!\left(\tfrac{q_t^\top k_s}{\sqrt d}\right) v_s$$

## Формы тензоров

```
x [B,T,C] → индексатор [B,T,H_I,S] → ReLU, Σ_h w → [B,T,S] (−inf будущее) → topk → keep [B,T,S] → attention [B,H,T,S] с ~keep → [B,T,C]
```

## Что поменялось относительно C07

Вместо фиксированного окна «последние W токенов» — своя маска у каждого запроса: top-k ключей по оценке
индексатора среди всего прошлого. При $k \ge T$ — обычное causal attention, как SWA при $W \ge T$.
