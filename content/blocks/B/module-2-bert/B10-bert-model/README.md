# Модель BERT: слой Post-LN с GELU, pooler, MLM-голова

## Зачем это нужно

BERT (Devlin et al., 2018) — это энкодер исходного Transformer из B03 без декодера, предобученный на MLM (B09).
Его новизна была не в архитектуре, а в рецепте: один большой энкодер предобучают на книгах и Википедии, а потом
с минимальной головой дообучают на любую задачу. Через год после выхода BERT был основой почти всех лучших
результатов в понимании текста. Многие энкодеры поиска и классификаторы в проде — его прямые потомки.

Ты собираешь BERT целиком, с именами параметров Hugging Face, и в конце загружаешь в **свою** модель настоящие
веса `bert-base-uncased`:

```
the capital of france is [MASK] .            → paris 0.42, lille 0.07, lyon 0.06
i drank a cup of hot [MASK] this morning .   → coffee 0.40, chocolate 0.27, water 0.13
```

А на русском, со словарём из B07, получается ерунда: `москва — столица [MASK] .` → `" ##я с ##т ##л` (каждый
вариант ~7 %). Модель английская.

## Что поменялось относительно B03

| | энкодер Transformer (B03) | BERT (B10) |
|---|---|---|
| вход | токен × √C + синусоиды | LN(токен + позиция + сегмент), B08 |
| слой | Post-LN, ReLU | Post-LN, **GELU** (точный, через erf) |
| имена | `self_attn`, `norm1`, `ffn.linear1` | `attention.self.query`, `attention.output.LayerNorm`, `intermediate.dense` |
| LayerNorm eps | 1e-5 | 1e-12 |
| выход | память для декодера | последовательность + **pooler** (`tanh(dense(h[CLS]))`) |
| голова | — | MLM: dense → GELU → LN → матрица эмбеддингов + bias |
| размер (base) | 6 + 6 слоёв, 512, 8 голов | 12 слоёв, 768, 12 голов, FFN 3072 — 109 М параметров |

Размеры — из [config.json](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json):
`hidden_size = 768`, `num_hidden_layers = 12`, `num_attention_heads = 12`, `intermediate_size = 3072`,
`max_position_embeddings = 512`, `vocab_size = 30522`, `layer_norm_eps = 1e-12`, `hidden_act = "gelu"`.

## Формула

Слой (Post-LN):

$$
h = \operatorname{LN}\big(x + W_O \operatorname{MHA}(x; M)\big), \qquad
y = \operatorname{LN}\big(h + W_2\, \operatorname{GELU}(W_1 h)\big),
\qquad \operatorname{GELU}(z) = z\,\Phi(z) = \tfrac{z}{2}\big(1 + \operatorname{erf}(z/\sqrt 2)\big).
$$

Pooler и MLM-голова:

$$
\text{pooled} = \tanh(W_p\, y_{[\text{CLS}]} + b_p), \qquad
\text{logits}_t = \operatorname{LN}\big(\operatorname{GELU}(W_t\, y_t + b_t)\big)\, E^{\top} + b_{\text{vocab}},
$$

где $E$ — матрица `word_embeddings`. Pooler обучался на задаче NSP (следует ли второй сегмент за первым).
RoBERTa от NSP отказалась, и для эмбеддингов текста pooler обычно не берут (модуль 4).

## Формы тензоров

```shapes
input_ids [B, T]
attention_mask [B, T] — 1 / 0
> bool, [:, None, None, :]
mask [B, 1, 1, T]
> embeddings (B08)
x [B, T, H]
## слой × 12
> attention.self: query / key / value, головы, attention (B01)
a [B, T, H]
> attention.output: dense, LN(x + ·)
h [B, T, H]
> intermediate.dense, GELU
f [B, T, 4H]
> output: dense, LN(h + ·)
x [B, T, H]
## конец стопки
fork
> pooler: tanh(dense(x[:, 0]))
pooled [B, H]
branch
> transform: dense → GELU → LN; decoder (= word_embeddingsᵀ) + bias
logits [B, T, V]
join
axis H = hidden_size (768 у base)
axis V:wide = 30 522
```

:::viz mlm-masking:::

## Задание

В `solution.py` (рабочая копия в `workspace/B10-bert-model/`) напиши `forward` четырёх классов:

1. `BertSelfAttention` — query / key / value, головы, `attention` из B01, склейка. Выходной проекции здесь нет.
2. `BertLayer` — Post-LN, точный GELU (`F.gelu` без `approximate`).
3. `BertModel` — маска из `attention_mask`, эмбеддинги, слои, pooler → `(sequence, pooled | None)`.
4. `BertForMaskedLM` — `bert` → transform → decoder.

Конструкторы готовы. `BertConfig` и `BertEmbeddings` — эталоны B08.

## Критерии готовности

- self-attention совпадает с ручным расчётом с маской;
- слой совпадает с эталоном при неединичных LayerNorm (Post-LN, точный GELU);
- `BertModel` возвращает `(sequence, pooled)`, `pooled = tanh(dense(h[CLS]))`, без pooler — `None`; паддинг не
  влияет на настоящие токены;
- логиты MLM совпадают с эталоном и с `transformers.BertForMaskedLM` на тех же случайных весах; декодер связан с
  эмбеддингами;
- маленький BERT за 150 шагов учится восстанавливать закрытый токен по соседям слева и справа;
- с `--weights`: настоящий `bert-base-uncased` в твоей модели даёт те же логиты, что `transformers`, и ставит
  «paris» на первое место.

Запуск: `llmlab test B10-bert-model` (с настоящими весами: `llmlab test B10-bert-model --weights`, ~440 МБ).
