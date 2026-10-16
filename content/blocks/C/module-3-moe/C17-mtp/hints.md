## Подсказка 1

`MTPModule.forward`: `x = self.eh_proj(torch.cat((self.enorm(emb_next), self.hnorm(h_prev)), dim=-1))`,
`return self.block(x)`. Порядок важен: сначала эмбеддинг.

## Подсказка 2

`mtp_logits`: цикл `for k, module in enumerate(modules, start=1)`. Держи переменную `h` (сначала `h_main`).
На шаге k: `h = module(h[:, :T - k], embed(tokens[:, k:]))` — у предыдущего `h` длина `T − k + 1`,
поэтому срез отбрасывает последнюю позицию. Логиты — `lm_head(module.norm(h))`.

## Подсказка 3

`mtp_loss`: `F.cross_entropy(logits[:, :-1].reshape(-1, V), targets.reshape(-1))`, где цели основной
головы — `tokens[:, 1:]`, глубины k — `tokens[:, k + 1:]`. Итог: `L_main + lam / D * sum(L_k)`.
