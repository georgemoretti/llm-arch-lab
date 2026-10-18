## Подсказка 1

`json.loads(Path(vocab_path).read_text(encoding="utf-8"))` — словарь `{строка: id}`. Для `merges.txt`:
`read_text(...).splitlines()`, затем выбрось пустые строки и заголовок (`line.startswith("#version")`).
Каждое правило — `a, b = line.split(" ")`: внутри строк токенов пробелов нет, пробел записан как `Ġ`.

## Подсказка 2

`str_to_ours = {ch: b for b, ch in bytes_to_unicode().items()}` — 256 строк-байтов. Правила идут в
порядке обучения, поэтому обе части $k$-го правила к моменту его чтения уже есть в `str_to_ours`: либо
это байт, либо результат более раннего правила. После правила добавь `str_to_ours[a + b] = 256 + k`.

## Подсказка 3

`to_gpt2 = {ours: gpt2_vocab[s] for s, ours in str_to_ours.items()}`, а спецтокены —
`{s: i for s, i in gpt2_vocab.items() if s not in str_to_ours}`. В `encode` повтори цикл
`encode_with_special` из A13 и пиши `ids.extend(self.to_gpt2[i] for i in encode(chunk, self.merges))`.
В `decode` байты id — `self.vocab[self.from_gpt2[i]]`, у спецтокена — `s.encode("utf-8")`.
