## Подсказка 1

```python
"".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
```

В `basic_tokenize` иди по символам каждого куска после `split()`. Накапливай буквы в `current`; на знаке
пунктуации сбрось `current` (если не пустой) и добавь сам знак отдельным словом.

## Подсказка 2

Жадный разбор — два вложенных цикла: внешний двигает `start`, внутренний уменьшает `end` от `len(word)`, пока
кусок `word[start:end]` (с `"##"` спереди, если `start > 0`) не найдётся в словаре. Если внутренний цикл дошёл до
`end == start`, разбор невозможен.

## Подсказка 3

```python
pieces, start = [], 0
while start < len(word):
    end = len(word)
    while end > start:
        piece = word[start:end] if start == 0 else "##" + word[start:end]
        if piece in vocab:
            break
        end -= 1
    if end == start:
        return [unk]
    pieces.append(piece)
    start = end
return pieces
```
