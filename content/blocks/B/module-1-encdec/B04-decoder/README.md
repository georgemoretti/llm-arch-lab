# Декодер Transformer и cross-attention

## Зачем это нужно

Энкодер прочитал исходную фразу и оставил память — по вектору на токен. Теперь перевод нужно **написать**,
токен за токеном. Пишет декодер. Он устроен как GPT из блока A: causal self-attention по уже написанному.
Но одного прошлого перевода мало: надо знать, что переводим. Для этого в каждом слое декодера есть третий
подслой — **cross-attention**. Запросы в нём идут из декодера («какое слово источника мне сейчас нужно?»),
а ключи и значения — из памяти энкодера.

Эту идею придумали до Transformer: Bahdanau et al. (2014) добавили attention к рекуррентному переводчику,
чтобы декодер не держал всю фразу в одном векторе. Transformer оставил от RNN только этот механизм, а
рекуррентность заменил self-attention.

Маски у трёх видов attention разные:

- **self-attention энкодера** — только padding источника (B01, B03);
- **self-attention декодера** — causal И padding цели: нельзя подсматривать следующие слова перевода;
- **cross-attention** — только padding источника. Causal здесь не нужен: источник известен целиком, а
  первое слово перевода часто зависит от конца фразы (в немецком глагол уходит в конец).

## Что поменялось относительно B03

| | слой энкодера (B03) | слой декодера (B04) |
|---|---|---|
| подслоёв | 2: self-attention, FFN | 3: masked self-attention, cross-attention, FFN |
| маска self-attention | padding $[B, 1, 1, S]$ | causal & padding $[B, 1, T, T]$ |
| ключи и значения | из своего входа | в cross-attention — из памяти энкодера $[B, S, C]$ |
| норм | `norm1`, `norm2` | `norm1`, `norm2`, `norm3` |

## Формула

Post-LN, как в статье:

$$
\begin{aligned}
h_1 &= \operatorname{LN}_1\big(y + \operatorname{MHA}(y, y;\; M_{\text{tgt}})\big),\\
h_2 &= \operatorname{LN}_2\big(h_1 + \operatorname{MHA}(h_1, \text{memory};\; M_{\text{src}})\big),\\
\text{out} &= \operatorname{LN}_3\big(h_2 + \operatorname{FFN}(h_2)\big),
\end{aligned}
$$

где $\operatorname{MHA}(a, b; M)$ — запросы из $a$, ключи и значения из $b$. Маски:

$$
M_{\text{tgt}}[b, t, s] = [s \le t] \wedge [\text{tgt}_{b,s} \ne \text{PAD}],
\qquad
M_{\text{src}}[b, \cdot, s] = [\text{src}_{b,s} \ne \text{PAD}].
$$

Матрица весов cross-attention прямоугольная: $T \times S$. Строка $t$ показывает, на какие слова источника
смотрит позиция $t$ перевода, — это «выравнивание» слов, которое видно в визуализации проекта B06.

## Формы тензоров

```shapes
tgt [B, T]
> target_mask: causal & padding
tgt_mask [B, 1, T, T]
> embed
y [B, T, C]
## слой × n_layers
> self_attn(y, tgt_mask) + y → norm1
h1 [B, T, C]
fork
- запросы
q [B, H, T, D]
branch
memory [B, S, C] — выход энкодера, общий для всех слоёв
> k_proj, v_proj
k, v [B, H, S, D]
join
> cross-attention с src_mask [B, 1, 1, S]
weights [B, H, T, S] — строка: позиция цели, столбец: токен источника
> out_proj + h1 → norm2 → FFN + остаток → norm3
y [B, T, C]
axis S = длина источника
axis T = длина цели
```

:::viz encdec-masks:::

## Задание

В `solution.py` (рабочая копия в `workspace/B04-decoder/`):

1. `causal_mask(T)` и `target_mask(tgt, pad_id)` → bool `[B, 1, T, T]`.
2. `DecoderLayer.forward(x, memory, tgt_mask, src_mask)`: три подслоя, Post-LN или Pre-LN (`norm_first`).
3. `Decoder.forward(tgt, memory, src_mask)`: маска цели, эмбеддинги, слои, `self.norm`.

Готовые `nn.TransformerDecoderLayer` и `nn.MultiheadAttention` запрещены: первый — оракул тестов.

## Критерии готовности

- `target_mask` — causal и padding по ключам цели;
- слой совпадает с `nn.TransformerDecoderLayer` с теми же весами в обоих режимах при $T \ne S$;
- каждая позиция цели, включая первую, зависит от последнего токена источника; PAD источника не влияет;
- стопка совпадает с эталоном; изменение цели после позиции $t$ не меняет выход на позициях $\le t$.

Запуск: `llmlab test B04-decoder`.
