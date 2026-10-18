## Подсказка 1

`BertSelfAttention.forward`:
```python
B, T, H = x.shape
split = lambda t: t.view(B, T, self.n_heads, H // self.n_heads).transpose(1, 2)
out, _ = attention(split(self.query(x)), split(self.key(x)), split(self.value(x)), mask)
return out.transpose(1, 2).reshape(B, T, H)
```

## Подсказка 2

`BertLayer.forward` — две строки «подслой, остаток, норма»:
```python
h = self.attention.output.LayerNorm(x + self.attention.output.dense(self.attention.self(x, mask)))
return self.output.LayerNorm(h + self.output.dense(F.gelu(self.intermediate.dense(h))))
```

## Подсказка 3

`BertModel.forward`: если `attention_mask is None` — `torch.ones_like(input_ids)`; маска
`attention_mask.bool()[:, None, None, :]`; `x = self.embeddings(input_ids, token_type_ids)`; цикл по
`self.encoder.layer`; `pooled = torch.tanh(self.pooler.dense(x[:, 0])) if self.pooler is not None else None`.

`BertForMaskedLM.forward`: `x, _ = self.bert(...)`, затем
`self.cls.predictions.decoder(t.LayerNorm(F.gelu(t.dense(x))))`, где `t = self.cls.predictions.transform`.
