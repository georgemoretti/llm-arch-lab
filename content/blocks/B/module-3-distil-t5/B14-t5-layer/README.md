# Слой T5: RMSNorm, Pre-LN, attention без 1/√d с относительным bias

## Зачем это нужно

T5 (Raffel et al., 2019) — энкодер-декодер исходного Transformer, собранный заново по итогам большого
сравнительного исследования. Все задачи в нём записаны как «текст → текст»: перевод, классификация («positive»),
регрессия (число строкой), суммаризация. Одна модель, один лосс, один формат входа с префиксом задачи
(`translate English to German: …`). Энкодер-декодеры этой линии (T5, Flan-T5, mT5, ByT5) до сих пор работают в проде
для перевода, суммаризации и извлечения данных, а энкодер T5 стоит текстовым энкодером в моделях «текст → картинка».

Устройство слоя T5 очень похоже на современные декодеры из блока C. T5 одной из первых убрала bias из линейных
слоёв, центрирование из нормы (RMSNorm, C01) и позиции из входа, а нормы поставила перед подслоями (Pre-LN, C08).
Ты собираешь T5 с именами Hugging Face и в конце загружаешь в **свою** модель настоящие веса `t5-small`:

```
The <extra_id_0> walks in <extra_id_1> park.   →   <extra_id_0> park offers <extra_id_1> the <extra_id_2> park.
```

Ответ странный, но формат верный: модель знает, что после sentinel идёт пропущенный кусок. У t5-small всего 60 М
параметров. Как устроены sentinel-токены, разберём в B15.

## Что поменялось относительно B04

| | слой декодера Transformer (B04) | T5 (B14) |
|---|---|---|
| норма | LayerNorm (центрирование + bias) | **T5LayerNorm** = RMSNorm: $x / \sqrt{\overline{x^2} + \epsilon} \cdot w$, eps 1e-6 |
| место нормы | Post-LN: `LN(x + sublayer(x))` | **Pre-LN**: `x + sublayer(LN(x))`, плюс `final_layer_norm` в конце стека |
| позиции | синусоиды на входе | нет на входе; **относительный bias** в scores (B13), таблица только у блока 0 |
| scores | $q \cdot k / \sqrt{D}$ | $q \cdot k + b$ — **без** $1/\sqrt{D}$ |
| bias линейных слоёв | есть | **нет** нигде: `q`, `k`, `v`, `o`, `wi`, `wo` |
| FFN | Linear → ReLU → Linear | `wo(ReLU(wi(x)))`; в T5 v1.1 — GeGLU (`wi_0`, `wi_1`) |
| эмбеддинги | свои у источника и цели | одна матрица `shared` у энкодера, декодера и головы |
| голова | Linear | `shared` с множителем $d_{\text{model}}^{-1/2}$ перед ней |

Почему нет $1/\sqrt{D}$: масштаб вынесен в инициализацию. Веса `q` в T5 инициализируют с дисперсией
$1/(d_{\text{model}} \cdot d_{kv})$, и дисперсия scores в начале обучения та же, что у обычного attention с делением.
Обученные веса уже рассчитаны на scores без деления. Поделишь на $\sqrt{D}$ — и t5-small выдаст мусор.

Размеры t5-small — из [config.json](https://huggingface.co/google-t5/t5-small/blob/main/config.json): `d_model = 512`,
`d_kv = 64`, `num_heads = 8`, `d_ff = 2048`, `num_layers = 6` (и 6 слоёв декодера), `vocab_size = 32128`,
`layer_norm_epsilon = 1e-6`, `relative_attention_num_buckets = 32`; `feed_forward_proj` не указан, по умолчанию —
`"relu"`. Всего 60 506 624 параметра. У [t5-v1_1-small](https://huggingface.co/google/t5-v1_1-small/blob/main/config.json):
`feed_forward_proj = "gated-gelu"`, `d_ff = 1024`, `num_layers = 8`, `num_heads = 6`, `tie_word_embeddings = false`.

## Формула

Блок энкодера и блок декодера (в декодере есть строка с cross-attention):

$$
\begin{aligned}
x &\leftarrow x + W_o\,\operatorname{softmax}\!\big(q k^{\top} + b_{\text{rel}} + M\big)\, v, && q, k, v = W_{q,k,v}\,\operatorname{RMS}_1(x) \\
x &\leftarrow x + W_o'\,\operatorname{softmax}\!\big(q' k'^{\top} + M_{\text{src}}\big)\, v', && q' = W_q'\,\operatorname{RMS}_2(x),\ k', v' = W_{k,v}'\, h_{\text{enc}} \\
x &\leftarrow x + W_{\text{wo}}\,\operatorname{ReLU}\!\big(W_{\text{wi}}\,\operatorname{RMS}_3(x)\big),
\end{aligned}
\qquad
\operatorname{RMS}(x) = \frac{x}{\sqrt{\tfrac{1}{C}\sum_c x_c^2 + \epsilon}} \odot w .
$$

Маски: в энкодере $M$ закрывает PAD-ключи; в self-attention декодера — PAD-ключи **и** будущее; в cross-attention
$M_{\text{src}}$ закрывает PAD источника. У cross-attention нет позиционного bias. Bias self-attention ($b_{\text{rel}}$)
считает блок 0 (двунаправленный в энкодере, causal в декодере), остальные блоки стека берут его же.

## Формы тензоров

```shapes
input_ids [B, S]
> shared (без позиций и масштаба)
x [B, S, C]
## блок энкодера × 6
> RMS → q, k, v → головы
q [B, H, S, D]
> q·kᵀ + bias (блок 0 считает, остальные берут) + маска PAD
scores [B, H, S, S]
> softmax @ v, склейка, o; x + ·
x [B, S, C]
> RMS → wi → ReLU → wo; x + ·
x [B, S, C]
## конец стопки
> final_layer_norm
memory [B, S, C]
## декодер: то же + cross-attention к memory
> q из декодера, k и v из memory, без bias
cross [B, H, T, S]
y [B, T, C]
axis C = d_model (512)
axis D = d_kv (64)
axis H = num_heads (8)
```

:::viz t5-buckets:::

## Задание

В `solution.py` (рабочая копия в `workspace/B14-t5-layer/`) напиши `forward` четырёх классов:

1. `T5LayerNorm` — RMSNorm, `mean(x²)` в fp32.
2. `T5Attention` — q / k / v из `x` (или k, v из `kv` для cross-attention), scores **без** $1/\sqrt{D}$, плюс
   `position_bias`: передан → как есть; не передан и есть таблица → `relative_position_bias` из B13 (энкодер —
   двунаправленный, декодер — causal); иначе нули. Вернуть `(выход, position_bias)`.
3. `T5Block` — три (у энкодера два) Pre-LN подслоя с остатками; вернуть `(x, position_bias)`.
4. `T5Stack` — маски, эмбеддинги, блоки с общим bias, финальная норма.

Конструкторы и `T5Model` готовы. `T5Config` и `relative_position_bias` — эталоны (B13).

## Критерии готовности

- T5LayerNorm не вычитает среднее, eps стоит под корнем, на fp16 с $|x| = 300$ не переполняется;
- attention: scores без $1/\sqrt{D}$, bias блока 0 = `relative_position_bias` из B13, переданный bias используется как
  есть, у слоя без таблицы — нули; cross-attention с $T \ne S$;
- блоки энкодера и декодера совпадают с эталоном при неединичных нормах (Pre-LN, ReLU, остатки, cross-attention);
- стек передаёт bias блока 0 следующим блокам и заканчивается `final_layer_norm`; паддинг не влияет на настоящие
  токены, декодер не видит будущего;
- выходы энкодера и декодера совпадают с `transformers.T5Model` на тех же случайных весах;
- с `--weights`: настоящий `t5-small` в твоей модели даёт те же логиты, что `transformers`, и отвечает в формате
  «sentinel + пропущенный кусок».

Запуск: `llmlab test B14-t5-layer` (с настоящими весами: `llmlab test B14-t5-layer --weights`, ~240 МБ).
