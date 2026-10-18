# Вход BERT: сегменты, обрезка пары, маска, три эмбеддинга

## Зачем это нужно

Много задач NLP устроено как «пара текстов»: вопрос и абзац с ответом (B11), два предложения — следует ли
второе из первого, запрос и документ для reranking. BERT обрабатывает пару **одной** последовательностью:

```
[CLS] как устроен bert ? [SEP] bert — это энкодер … [SEP]
  0    0     0      0   0   0    1    1   1   1     1     ← token_type_ids (сегмент)
```

- `[CLS]` стоит в начале. Его выход — вектор «всего входа»: по нему классифицируют (pooler, B10–B11);
- `[SEP]` закрывает каждый сегмент;
- `token_type_ids` говорят, какому сегменту принадлежит токен. У BERT для них есть третья таблица
  эмбеддингов: вектор сегмента прибавляется к каждому токену. Так модель отличает вопрос от абзаца.

Длина ограничена (512 у `bert-base-uncased`, `max_position_embeddings` в config.json), поэтому длинную пару
обрезают. Стратегия `longest_first` старается сохранить оба сегмента: короткий не трогают, если он не
длиннее половины бюджета, иначе бюджет делится пополам. Тексты разной длины в батче выравнивают PAD, а
`attention_mask` (1 — токен, 0 — PAD) превратится в padding-маску по ключам из B01.

## Что поменялось относительно B02

| | вход Transformer (B02) | вход BERT (B08) |
|---|---|---|
| позиции | синусоиды, не обучаются | обучаемая таблица `position_embeddings` (как у GPT в A15), до 512 |
| сегменты | нет: источник и цель — разные стеки | `token_type_embeddings`, вход — пара в одной последовательности |
| масштаб | токены × $\sqrt C$ | без множителя, сумма сразу идёт в LayerNorm |
| служебные токены | BOS, EOS, PAD | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` |

## Формула

$$
x_t = \operatorname{LayerNorm}\big(E_{\text{word}}[\text{id}_t] + E_{\text{pos}}[t] + E_{\text{type}}[\text{type}_t]\big),
\qquad t = 0 \dots T - 1.
$$

Обрезка пары длин $n_a, n_b$ под бюджет $M = \text{max\_len} - 3$ (три служебных токена); длинный — тот, что
длиннее (при равенстве — $b$):

$$
n_{\text{short}}' = \begin{cases} n_{\text{short}}, & n_{\text{short}} \le \lfloor M/2 \rfloor \\
\lfloor M/2 \rfloor, & \text{иначе} \end{cases}
\qquad
n_{\text{long}}' = M - n_{\text{short}}'.
$$

Исходный код BERT (2018) резал иначе: по одному токену у более длинного сегмента, пока не влезет. Результат
почти всегда тот же, но при равных длинах отличается. Здесь — правило быстрых токенизаторов Hugging Face,
которыми сейчас пользуются все.

## Формы тензоров

```shapes
- пара id: a [n_a], b [n_b]
> build_inputs: обрезка, [CLS] a [SEP] b [SEP]
ids [≤ max_len]
> pad_batch
input_ids [B, T]
token_type_ids [B, T] — 0 / 1
attention_mask [B, T] — 1 токен, 0 PAD
fork
> word_embeddings
w [B, T, H]
branch
> position_embeddings(0 … T − 1)
p [T, H]
branch
> token_type_embeddings
s [B, T, H]
join
> сумма → LayerNorm (eps = 1e-12) → dropout
x [B, T, H]
axis H = hidden_size
```

:::viz wordpiece:::

## Задание

В `solution.py` (рабочая копия в `workspace/B08-bert-input/`):

1. `truncate_pair(a, b, max_tokens)` — обрезка `longest_first`, входные списки не меняются.
2. `build_inputs(a, b, cls_id, sep_id, max_len)` → `(input_ids, token_type_ids)`; `b=None` — один сегмент.
3. `pad_batch(examples, pad_id)` → словарь из трёх тензоров `long [B, T_max]`.
4. `BertEmbeddings.forward(input_ids, token_type_ids=None)`. Конструктор готов: имена — как в Hugging Face.

## Критерии готовности

- обрезка совпадает с библиотекой `tokenizers` (`longest_first`) для всех длин сегментов 0–13 и бюджетов 3–10;
- `[CLS] a [SEP] b [SEP]`, типы 0 у первого сегмента с его `[SEP]` и 1 у второго; длина не больше `max_len`;
- батч: PAD справа, тип 0 и маска 0 на PAD;
- эмбеддинги совпадают с эталоном и с `transformers` `BertEmbeddings` на тех же весах; без `token_type_ids` —
  сегмент 0.

Запуск: `llmlab test B08-bert-input`.
