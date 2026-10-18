## Подсказка 1

```python
m = attention_mask.to(hidden.dtype).unsqueeze(-1)          # [B, T, 1]
(hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
```
CLS: `hidden[:, 0]`.

## Подсказка 2

`l2_normalize`: `x / x.norm(dim=-1, keepdim=True).clamp(min=eps)`.
`cosine_matrix`: `l2_normalize(a) @ l2_normalize(b).T`.

## Подсказка 3

```python
h = _hidden(self.encoder, input_ids, attention_mask)
e = mean_pool(h, attention_mask) if self.pooling == "mean" else cls_pool(h)
return l2_normalize(e) if self.normalize else e
```
Cross-encoder: `h = _hidden(self.encoder, input_ids, attention_mask, token_type_ids)`;
`self.classifier(cls_pool(h)).squeeze(-1)`.
