## Подсказка 1

`SequenceClassifier.forward`: `_, pooled = self.bert(input_ids, token_type_ids, attention_mask)`, затем
`self.classifier(self.dropout(pooled))`.

`align_labels`: иди по `word_ids`, помни предыдущий `word_id`. Токен — первый в слове, если его `word_id` не `None`
и отличается от предыдущего.

## Подсказка 2

`qa_loss`: `(F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2`.

`best_span`: `scores = start_logits[:, :, None] + end_logits[:, None, :]` — строка $i$ (начало), столбец $j$ (конец).

## Подсказка 3

```python
i = torch.arange(T)
allowed = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < max_answer_len)   # [T, T]: j ≥ i, j − i < L
allowed = allowed & context_mask[:, :, None] & context_mask[:, None, :]           # [B, T, T]
flat = scores.masked_fill(~allowed, float("-inf")).view(B, -1).argmax(-1)
return flat // T, flat % T
```
