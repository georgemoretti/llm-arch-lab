## Подсказка 1

`distill_kl` удобно считать в логарифмах — так нет `log(0)`:
```python
V = student_logits.shape[-1]
log_q = F.log_softmax(student_logits.reshape(-1, V) / T, dim=-1)
log_p = F.log_softmax(teacher_logits.detach().reshape(-1, V) / T, dim=-1)
kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
return kl * T**2
```
То же самое: `F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")`. Осторожно: у `F.kl_div` первым идёт
**ученик** (логарифм), вторым — учитель, а `reduction="mean"` делит ещё и на число классов.

## Подсказка 2

`distillation_loss`: `soft = distill_kl(...)`; `hard = F.cross_entropy(s.reshape(-1, V), labels.reshape(-1),
ignore_index=-100)`, но только если `(labels != -100).any()`, иначе `hard = 0` (среднее по пустому множеству — nan).

`hidden_cosine_loss`: `m = attention_mask.bool()`; `cos = F.cosine_similarity(hs[m], ht[m].detach(), dim=-1)`;
`return (1 - cos).mean()`. Индексация bool-маской `[B, T]` превращает `[B, T, H]` в `[N, H]`.

## Подсказка 3

`student_from_teacher`:
```python
n = len(teacher.bert.encoder.layer)
layers = list(range(0, n, 2)) if layers is None else list(layers)
student = BertForMaskedLM(replace(teacher.bert.cfg, num_hidden_layers=len(layers)))
student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())
for i, j in enumerate(layers):
    student.bert.encoder.layer[i].load_state_dict(teacher.bert.encoder.layer[j].state_dict())
student.cls.load_state_dict(teacher.cls.state_dict())
```
`load_state_dict` копирует значения в собственные тензоры ученика, и связь «декодер = эмбеддинги» у него сохраняется.
