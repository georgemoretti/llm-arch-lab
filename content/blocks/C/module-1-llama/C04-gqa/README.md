# Grouped-Query Attention (GQA и MQA)

## Зачем это нужно

При генерации модель выдаёт по одному токену, и чтобы не пересчитывать ключи и значения всех
предыдущих токенов на каждом шаге, их хранят в **KV-cache** (его соберём в C05). Размер кеша:

$$
\text{bytes} = 2 \cdot L \cdot H_{kv} \cdot D \cdot S \cdot B \cdot \text{bytes\_per\_elem}
$$

(2 — это K и V; $L$ — слои, $H_{kv}$ — KV-головы, $D$ — размер головы, $S$ — длина контекста,
$B$ — batch). У Llama 2 7B с обычным multi-head attention это 0,5 МиБ на **каждый** токен:
контекст 4096 — 2 ГиБ, а batch 16 — уже 32 ГиБ, больше, чем весят сами веса (≈ 12,5 ГиБ в fp16).
Причём на каждом шаге генерации весь этот кеш надо прочитать из памяти, так что декодирование
упирается не в вычисления, а в пропускную способность памяти.

Наблюдение: головам запросов не обязательно иметь **собственные** K и V.

- **MQA** (multi-query attention, Shazeer, 2019): все головы запросов делят одну пару K/V.
  Кеш меньше в $H$ раз, но качество заметно падает и обучение менее стабильно.
- **GQA** (grouped-query attention, Ainslie et al., 2023): головы запросов разбиты на $H_{kv}$
  групп, у каждой группы своя пара K/V. MHA ($H_{kv} = H$) и MQA ($H_{kv} = 1$) — крайние случаи.
  Качество почти как у MHA, кеш меньше в $H / H_{kv}$ раз.

:::viz kv-cache:::

## Что поменялось относительно GPT-2

| | GPT-2 (MHA) | Llama 3 (GQA) |
|---|---|---|
| проекции | одна `c_attn`: C → 3C | `q_proj`: C → H·D, `k_proj`, `v_proj`: C → **H_kv**·D, `o_proj` |
| K/V-головы | по одной на каждую голову запросов | одна на группу из H / H_kv голов |
| bias | есть | нет |
| KV-cache на токен | 2 · L · H · D | 2 · L · **H_kv** · D — в H / H_kv раз меньше |
| сама формула attention | softmax(QKᵀ/√D + M)V | та же; меняется только, откуда голова берёт K и V |

Llama 3 8B: 32 головы запросов, 8 KV-голов, `head_dim` = 4096 / 32 = 128, т. е. группы по 4
(config.json `meta-llama/Meta-Llama-3-8B`: `num_attention_heads = 32`, `num_key_value_heads = 8`;
сверено по зеркалу `NousResearch/Meta-Llama-3-8B`). У Llama 2 7B `num_key_value_heads = 32` — это
ещё MHA; GQA в Llama 2 был только у 70B (64 головы, 8 KV-голов).

## Формула

Голова запросов $h \in \{0, \dots, H-1\}$ относится к группе $g(h) = \lfloor h / r \rfloor$, где
$r = H / H_{kv}$ — размер группы (`n_rep`):

$$
\text{head}_h = \operatorname{softmax}\!\left(\frac{Q_h K_{g(h)}^\top}{\sqrt{D}} + M\right) V_{g(h)}
$$

$M$ — causal-маска, как в C00. Реализация «в лоб»: размножить каждую KV-голову $r$ раз так,
чтобы получилось $H$ голов, и дальше считать обычный MHA. Важно, **как** размножить: копии одной
KV-головы должны идти подряд — `[k0, k0, k1, k1]`, тогда голова $h$ получит $K_{\lfloor h/r \rfloor}$.

## Формы тензоров

```shapes
x [B, T, C]
fork
> q_proj → view(B, T, H, D) → transpose(1, 2)
q [B, H, T, D]
branch
> k_proj / v_proj → view(B, T, H_kv, D) → transpose(1, 2)
k, v [B, H_kv, T, D] — именно это хранит KV-cache
> repeat_kv(·, r): [B, H_kv, T, D] → [B, H_kv, r, T, D] → reshape
k, v [B, H, T, D] — головы 0..r−1 — копии KV-головы 0, и т. д.
join q·kᵀ/√D, маска, softmax, @ v
out [B, H, T, D]
> transpose(1, 2) → reshape(B, T, H·D) → o_proj
y [B, T, C]
axis r = H / H_kv — сколько голов запросов делят одну KV-голову
```

`repeat_kv` здесь — для простоты. Быстрые ядра (FlashAttention и т. п.) не копируют K/V, а
индексируют общую голову напрямую; результат тот же.

:::viz gqa-groups:::

## Задание

В `solution.py` (рабочая копия в `workspace/C04-gqa/`) реализуй:

1. `repeat_kv(x, n_rep)`: `[B, H_kv, T, D] → [B, H_kv·n_rep, T, D]`, копии подряд;
2. `kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_elem=2)` — по
   формуле выше;
3. `GroupedQueryAttention.forward(x)` — causal GQA. Конструктор дан, имена `q_proj`, `k_proj`,
   `v_proj`, `o_proj` — как в Hugging Face, не меняй их.

RoPE здесь пока не нужен — его добавим в C05. `F.scaled_dot_product_attention` запрещён.

## Критерии готовности

- `repeat_kv` даёт правильную форму, копии одной головы идут подряд, при `n_rep = 1` ничего не меняет;
- `kv_cache_bytes`: Llama 2 7B на 4096 токенах в fp16 — 2 ГиБ, Llama 3 8B на 8192 токенах — 1 ГиБ;
- `k_proj`/`v_proj` уже `q_proj` в `n_rep` раз, bias нет; выход `[B, T, C]`;
- выход совпадает с оракулом для GQA, MHA ($H_{kv} = H$) и MQA ($H_{kv} = 1$);
- GQA равен MHA, у которого веса K/V внутри группы одинаковые;
- будущее не протекает, примеры в батче независимы.

Запуск: `llmlab test C04-gqa`.
