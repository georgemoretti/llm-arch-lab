## Подсказка 1

Регулярку удобно собирать из строк таблицы через `|`, ровно в том порядке:
`r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ..."""`. Тройные кавычки нужны, потому что в шаблоне есть
апостроф. Проверяй на примерах из «Критериев»: `regex.findall(GPT2_PATTERN, "a  b")`.

## Подсказка 2

`bytes_to_unicode`: `visible = list(range(ord("!"), ord("~") + 1)) + …` — три диапазона включительно.
Потом один проход `for b in range(256)`: видимому байту — `chr(b)`, остальным — `chr(256 + n)`, где `n`
увеличивается на 1 для каждого невидимого байта.

## Подсказка 3

`split_special`: `pattern = "(" + "|".join(regex.escape(s) for s in special) + ")"`, затем
`[p for p in regex.split(pattern, text) if p]`. Скобки — чтобы `split` вернул и сами спецтокены.
`encode_with_special`: для каждой части — либо `ids.append(special[part])`, либо
`for chunk in pretokenize(part): ids.extend(encode(chunk, merges))`. В `decode_with_special` собери
байты всех id (`special_str.encode("utf-8")` для спецтокенов) и декодируй один раз.
