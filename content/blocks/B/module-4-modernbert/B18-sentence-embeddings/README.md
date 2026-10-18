# Sentence-эмбеддинги: mean / CLS pooling, L2, косинус; bi-encoder и cross-encoder

## Зачем это нужно

Энкодер выдаёт вектор на **каждый токен** `[B, T, C]`. Для поиска, кластеризации и дедупликации нужен один вектор на
**весь текст**: такие векторы можно заранее посчитать для миллиона документов, положить в индекс и искать ближайшие к
запросу по косинусу. Так устроены все модели эмбеддингов, от Sentence-BERT до E5 и BGE. Это **bi-encoder**: запрос и
документ кодируются независимо.

Обычный BERT сравнивает два текста иначе: склеивает их в один вход `[CLS] q [SEP] d [SEP]` и смотрит на `[CLS]` (B11).
Такая модель называется **cross-encoder**. Она точнее, потому что токены запроса видят токены документа в каждом слое.
Но для каждой пары её нужно прогонять заново. Найти самую похожую пару среди 10 000 предложений cross-encoder'ом — это
≈ 50 млн прогонов BERT, по оценке авторов Sentence-BERT около 65 часов. Bi-encoder обходится 10 000 прогонами и
матрицей косинусов, около 5 секунд. Поэтому в проде их ставят друг за другом: bi-encoder находит сотню кандидатов,
cross-encoder (reranker) переставляет их.

## Что поменялось относительно B11

| | классификатор пары (B11) | bi-encoder (B18) | cross-encoder (B18) |
|---|---|---|---|
| вход | `[CLS] a [SEP] b [SEP]` | каждый текст отдельно | `[CLS] q [SEP] d [SEP]` |
| выход | логиты классов по pooler | вектор текста `[C]`, норма 1 | одна оценка пары |
| из чего вектор | `tanh(dense(h[CLS]))` | среднее по токенам (mean) или `h[CLS]` | `h[CLS]` |
| можно заранее посчитать документы | нет | **да** | нет |
| цена поиска по $M$ документам | — | 1 прогон + $M$ скалярных произведений | $M$ прогонов |

**Mean или CLS.** Какой pooling у модели, решили при её обучении: какой вектор учили сближать, тот и надо брать. У
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — mean, у
[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — CLS (оба: BERT на 384, после pooling — L2; MiniLM
— 6 слоёв, bge-small — 12). Это записано в `1_Pooling/config.json` модели. Перепутаешь — качество поиска упадёт, хотя
код отработает без ошибок.

**Почему mean pooling считают по маске.** Если усреднять и PAD, эмбеддинг короткого текста зависит от того, с какими
текстами он попал в батч. Один и тот же документ в индексе и в запросе получит разные векторы.

## Формула

$$
e_{\text{mean}} = \frac{\sum_t m_t\, h_t}{\max\!\big(\sum_t m_t,\ \varepsilon\big)}, \qquad
e_{\text{cls}} = h_0, \qquad
\hat e = \frac{e}{\max(\lVert e \rVert, \varepsilon)}, \qquad
\cos(a, b) = \hat a \cdot \hat b .
$$

Матрица косинусов $N$ запросов и $M$ документов — одно умножение: $\hat A \hat B^{\top}$, форма $[N, M]$.
Cross-encoder: $s(q, d) = w^{\top} h_0\big([\text{CLS}]\, q\, [\text{SEP}]\, d\, [\text{SEP}]\big) + b$.

## Формы тензоров

```shapes
input_ids [B, T]
> энкодер (ModernBERT B16 или BERT B10)
h [B, T, C]
> mean по маске или h[:, 0]
e [B, C]
> L2
e [B, C]
## поиск
> запросы [N, C] × документы [M, C]: Â · B̂ᵀ
cos [N, M]
## cross-encoder
> вход-пара [CLS] q [SEP] d [SEP] → энкодер → h[:, 0] → Linear
s [B]
```

:::viz embedding-space:::

## Задание

В `solution.py` (рабочая копия в `workspace/B18-sentence-embeddings/`) напиши:

1. `mean_pool(hidden, attention_mask)` и `cls_pool(hidden)`.
2. `l2_normalize(x)` (руками, без `F.normalize`) и `cosine_matrix(a, b)`.
3. `BiEncoder.forward` — энкодер → pooling → L2.
4. `CrossEncoder.forward` — энкодер на паре → `[CLS]` → `classifier` → `[B]`.

`_hidden` готова: она достаёт `[B, T, C]` и из `ModernBertModel`, и из `BertModel` (тот возвращает пару).

## Критерии готовности

- mean pooling не зависит от значений на PAD и делит на число настоящих токенов; пустой текст даёт нули, а не nan;
- L2: норма 1, ноль остаётся нулём; косинусы `[N, M]`;
- эмбеддинг текста не зависит от длины паддинга и соседей по батчу (ModernBERT и BERT);
- `pooling="mean" | "cls"`, `normalize=False` оставляет длину;
- cross-encoder: одна оценка на пару, `token_type_ids` доходят до BERT, паддинг не влияет;
- с `--weights`: all-MiniLM-L6-v2 и bge-small-en-v1.5 в `BertModel` курса дают те же эмбеддинги, что `transformers`, и
  парафразы у них ближе друг к другу, чем к текстам на чужую тему.

Запуск: `llmlab test B18-sentence-embeddings` (с весами: `--weights`, ~90 + 130 МБ).
