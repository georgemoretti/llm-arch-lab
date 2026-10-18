# Энкодер Transformer: слой Post-LN и стопка

## Зачем это нужно

Энкодер превращает исходный текст в **память**: по вектору на каждый токен, и каждый вектор уже знает
контекст всего предложения. Дальше эту память читает декодер (B04), а в BERT (модуль 2) она сама становится
результатом: по ней классифицируют, размечают и ищут.

Слой энкодера почти совпадает с блоком из A17: attention, FFN, остаточные связи, LayerNorm. Различий три:
attention двунаправленный, с padding-маской из B01; FFN с ReLU; и LayerNorm стоит **после** сложения с
остатком (Post-LN). В A17 ты сравнивал pre-LN и post-LN на одном блоке. Здесь видно, откуда взялся post-LN:
так было в исходной статье. Позже почти все перешли на Pre-LN: глубокие Post-LN сети без долгого прогрева
learning rate обучаются плохо (Xiong et al., 2020). Поэтому слой умеет оба режима.

## Что поменялось относительно A17

| | A17 (блок GPT) | B03 (слой энкодера) |
|---|---|---|
| attention | одна голова, causal | multi-head, padding-маска, видит весь текст |
| FFN | $C \to 4C$, GELU | $C \to d_{ff}$, ReLU (в статье $d_{ff} = 2048$ при $C = 512$) |
| нормы | Pre-LN по умолчанию | Post-LN по умолчанию (`norm_first=False`) |
| имена | `ln_1`, `attn`, `mlp` | `norm1`, `self_attn`, `ffn` — как в `nn.TransformerEncoderLayer` |

## Формула

Post-LN (статья):

$$
h = \operatorname{LN}_1\big(x + \operatorname{MHA}(x)\big), \qquad y = \operatorname{LN}_2\big(h + \operatorname{FFN}(h)\big).
$$

Pre-LN:

$$
h = x + \operatorname{MHA}(\operatorname{LN}_1 x), \qquad y = h + \operatorname{FFN}(\operatorname{LN}_2 h),
$$

и после последнего слоя стопки — ещё один $\operatorname{LN}$: иначе выход энкодера не нормирован. У Post-LN
выход каждого слоя уже прошёл LayerNorm, финальная норма не нужна.

$$
\operatorname{FFN}(x) = \max(0,\; x W_1 + b_1)\, W_2 + b_2.
$$

## Формы тензоров

```shapes
src [B, S]
> padding_mask (одна на все слои)
mask [B, 1, 1, S]
> embed: tok · √C + PE
x [B, S, C]
## слой × n_layers
> self_attn(x, mask) + x → norm1
h [B, S, C]
> linear1, ReLU
h1 [B, S, d_ff]
> linear2 + h → norm2
x [B, S, C]
## конец стопки
> norm (Pre-LN) или Identity (Post-LN)
memory [B, S, C]
axis d_ff:wide = ширина FFN
```

:::viz encdec-masks:::

## Задание

В `solution.py` (рабочая копия в `workspace/B03-encoder/`):

1. `FeedForward.forward`: `linear1` → ReLU → `linear2`.
2. `EncoderLayer.forward(x, mask)`: Post-LN при `norm_first=False`, Pre-LN при `True`.
3. `Encoder.forward(src)`: маска по `src`, эмбеддинги, все слои с одной маской, `self.norm`.

`MultiHeadAttention`, `TransformerEmbedding` и `padding_mask` импортированы из эталонов B01–B02. Готовые
`nn.TransformerEncoderLayer` и `nn.MultiheadAttention` запрещены: первый — оракул тестов.

## Критерии готовности

- FFN с ReLU;
- слой совпадает с `nn.TransformerEncoderLayer` с теми же весами в обоих режимах (LayerNorm с неединичными
  весами, батч с паддингом);
- у Post-LN выход слоя нормирован (среднее 0, дисперсия 1 по $C$);
- стопка совпадает с эталоном в обоих режимах; лишние PAD не меняют память настоящих токенов; позиция 0
  зависит от последнего токена.

Запуск: `llmlab test B03-encoder`.
