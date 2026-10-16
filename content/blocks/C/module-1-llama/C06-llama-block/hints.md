## Подсказка 1

Посмотри, как устроен `Block` в `llmlab/core/gpt2.py`: структура та же, меняются только классы и
имена. Конструктор блока: `RMSNorm(cfg.d_model, cfg.rms_eps)` дважды,
`LlamaAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.rope_theta)`,
`SwiGLUFFN(cfg.d_model, cfg.d_hidden)`.

## Подсказка 2

Pre-Norm: `x = x + sublayer(norm(x))`. Не `h = norm(x); x = h + sublayer(h)` — тогда остаточный
поток каждый раз нормируется, и блок с нулевыми подслоями уже не тождественный. Кеш передаётся
только в attention: `self.self_attn(self.input_layernorm(x), cache)`.

## Подсказка 3

Модель: `nn.ModuleList(LlamaBlock(cfg) for _ in range(cfg.n_layers))` — именно `ModuleList`,
иначе параметры слоёв не попадут в `state_dict`. В `forward`:
`layer(x, caches[i] if caches is not None else None)`. Связывание весов:
`self.lm_head.weight = self.embed_tokens.weight` — присваивание объекта, а не копирование данных.
