# Слой ModernBERT: RoPE, GeGLU, без bias, глобальное и локальное attention

## Зачем это нужно

BERT (B10) вышел в 2018 году, и шесть лет его архитектура почти не менялась: в проде для классификации, разметки и
поиска стояли BERT, RoBERTa, DeBERTa с контекстом 512 токенов. Тем временем декодеры (блок C) получили RoPE, GLU-FFN,
Pre-LN, линейные слои без bias и чередование локального и глобального attention. ModernBERT (Warner et al., 2024)
переносит эти приёмы в энкодер. Получился BERT с контекстом 8192 токена: на длинных текстах он заметно быстрее старых энкодеров, а по качеству
на классификации и поиске их обгоняет (таблицы — в B-P4).

Почти все детали ты уже встречал в блоке C. Новое здесь одно: окно локального attention **двунаправленное**, ведь
энкодер смотрит и вправо. В конце задачи ты загрузишь в **свою** модель настоящие веса ModernBERT-base и заполнишь
`[MASK]`:

```
The capital of France is [MASK].   →   Paris
```

## Что поменялось относительно B10

| | BERT (B10) | ModernBERT (B16) |
|---|---|---|
| позиции | обучаемые `position_embeddings`, максимум 512 | **RoPE** в каждом слое (C03); θ = 160 000 в глобальных слоях, 10 000 в локальных |
| attention | все слои видят весь текст | **глобальный слой через два локальных**: слой 0, 3, 6, … — весь текст, остальные — окно ±64 токена |
| норма | Post-LN: `LN(x + sublayer(x))` | **Pre-LN**: `x + sublayer(LN(x))`, плюс `final_norm`; LayerNorm без bias |
| нормы у слоя 0 | — | `attn_norm = Identity`: вход уже нормирован LayerNorm эмбеддингов |
| FFN | Linear → GELU → Linear | **GeGLU**: `Wo(GELU(a) · g)`, где `[a | g] = Wi(x)` |
| bias | везде | **нет** ни в одном линейном слое и ни в одной норме (кроме bias декодера MLM-головы) |
| q, k, v | три матрицы `query`, `key`, `value` | одна `Wqkv` на `3·C` выходов |
| token type | есть (две части входа) | нет |
| контекст | 512 | 8192 |

Размеры ModernBERT-base — из [config.json](https://huggingface.co/answerdotai/ModernBERT-base/blob/main/config.json):
`hidden_size = 768`, `num_hidden_layers = 22`, `num_attention_heads = 12` (D = 64), `intermediate_size = 1152` (у `Wi`
выходов вдвое больше, 2304), `global_attn_every_n_layers = 3`, `local_attention = 128`, `global_rope_theta = 160000`,
`local_rope_theta = 10000`, `norm_eps = 1e-5`, `vocab_size = 50368`, `max_position_embeddings = 8192`. У
[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large/blob/main/config.json): 1024, 28 слоёв, 16 голов,
`intermediate_size = 2624`, окно и θ те же.

Почему окно не ломает качество. В локальном слое запрос видит 129 ключей, а не 8192. На длинном тексте это экономит
почти всю квадратичную часть вычислений. Дальние связи переносят глобальные слои, а за три слоя (глобальный и два
локальных) информация успевает разойтись по тексту. Тот же расчёт был в Gemma (`C07`), только там окно смотрит
назад, а здесь в обе стороны.

## Формула

$$
\begin{aligned}
x_0 &= \operatorname{LN}(E[\text{ids}]), \\
x &\leftarrow x + W_o\,\operatorname{softmax}\!\Big(\tfrac{R_\theta q\,(R_\theta k)^{\top}}{\sqrt{D}} + M_\ell\Big)\,v,
  && [q\,|\,k\,|\,v] = W_{qkv}\,\operatorname{LN}_{\text{attn}}(x), \\
x &\leftarrow x + W_o'\big(\operatorname{GELU}(a) \odot g\big), && [a\,|\,g] = W_i\,\operatorname{LN}_{\text{mlp}}(x), \\
h &= \operatorname{LN}_{\text{final}}(x).
\end{aligned}
$$

$R_\theta$ — поворот RoPE, $\theta$ своя у типа слоя. Маска $M_\ell$: в глобальном слое закрыты PAD-ключи, в
локальном — PAD-ключи **и** ключи дальше половины окна:

$$
\text{локальный слой: } j \text{ виден из } i \iff \text{mask}_j = 1 \ \wedge\ |i - j| \le \lfloor W / 2 \rfloor .
$$

Закрытые ключи заполняем `torch.finfo(dtype).min`, а не $-\infty$. В локальном слое PAD-запрос в конце длинного
хвоста может не увидеть ни одного ключа: при $-\infty$ softmax делит 0 на 0, и nan расходится дальше по слоям.

## Формы тензоров

```shapes
input_ids [B, T]
> tok_embeddings → LayerNorm (без позиций)
x [B, T, C]
## слой × 22: глобальный (0, 3, 6, …) или локальный
> attn_norm (у слоя 0 — Identity) → Wqkv
qkv [B, T, 3·C]
> view(B, T, 3, H, D) → permute
q [B, H, T, D]
> RoPE (θ слоя) для q и k; q·kᵀ / √D + маска (PAD; в локальном ещё |i − j| ≤ 64)
scores [B, H, T, T]
> softmax @ v → склейка → Wo; x + ·
x [B, T, C]
> mlp_norm → Wi
ag [B, T, 2·I]
> GELU(a) · g → Wo; x + ·
x [B, T, C]
## конец стопки
> final_norm
h [B, T, C]
axis C = hidden_size (768)
axis H = num_attention_heads (12)
axis D = C / H (64)
axis I = intermediate_size (1152)
```

## Задание

В `solution.py` (рабочая копия в `workspace/B16-modernbert-layer/`) напиши по порядку:

1. `layer_types(num_layers, global_every)` — список `"full_attention"` / `"sliding_attention"`. Начни с неё: её
   вызывает конструктор attention.
2. `local_window_mask(T, window)` — bool `[T, T]`, `|i − j| ≤ window // 2`.
3. `ModernBertMLP.forward` — GeGLU.
4. `ModernBertAttention.forward` — `Wqkv`, RoPE с `self.rope_theta`, $1/\sqrt{D}$, маска через `finfo.min`, `Wo`.
5. `ModernBertEncoderLayer.forward` — два Pre-LN подслоя с остатками.
6. `ModernBertModel.forward` — маска PAD, маска окна для локальных слоёв, эмбеддинги, слои, `final_norm`.

Конструкторы, эмбеддинги и `ModernBertForMaskedLM` готовы. RoPE берётся готовым из C03: `rope_cos_sin`, `apply_rope`.

## Критерии готовности

- глобальные слои 0, 3, 6, …; окно ±W/2, симметричное, граница включена;
- GeGLU: GELU на первой половине `Wi`, вторая — gate;
- attention: RoPE с θ своего слоя, деление на $\sqrt{D}$, правильная нарезка `Wqkv`, нет nan у запроса без
  видимых ключей;
- слой: Pre-LN, остатки мимо нормы;
- модель: локальные слои видят только окно, PAD не влияет на настоящие токены, в конце `final_norm`;
- выход совпадает с `transformers.ModernBertModel` на тех же случайных весах;
- с `--weights`: ModernBERT-base в твоей модели даёт те же логиты, что `transformers`, и заполняет `[MASK]` словом
  «Paris».

Запуск: `llmlab test B16-modernbert-layer` (с настоящими весами: `llmlab test B16-modernbert-layer --weights`, ~600 МБ).
