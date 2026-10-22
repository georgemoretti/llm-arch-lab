## Подсказка 1

`build_example` — три готовых шага из D01 и D02:
```python
ids = encode_chat(render_chat(to_messages(example)), tok)
mask = assistant_mask(ids, assistant_header(tok), tok.eos_id)     # или torch.ones(len(ids), dtype=torch.bool)
return ids, sft_targets(ids, mask)
```

## Подсказка 2

В `check_answer` JSON сравнивай как объекты: `json.loads(answer) == json.loads(want)`. Тогда ответ
`{"age":76,"name":"Педу"}` верный, а `{"name": "Педу", "age": "76"}` — нет: возраст строкой не проходит проверку
формата. Учти, что `isinstance(True, int)` в Python тоже `True`.

## Подсказка 3

```python
text = render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
ids = encode_chat(text, tok)
out = generate(model, torch.tensor([ids]), max_new_tokens, eos_token_id=tok.eos_id)[0, len(ids):].tolist()
if tok.eos_id in out:
    out = out[:out.index(tok.eos_id)]
return tok.decode(out)
```
Пустой блок `<think>` в промпте нужен потому, что в обучающем примере последний ответ получает его по правилу D01.
