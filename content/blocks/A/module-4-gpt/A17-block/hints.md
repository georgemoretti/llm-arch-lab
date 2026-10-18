## Подсказка 1

`MLP.forward`: `self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))`. Точный GELU (`approximate="none"`,
по умолчанию) отличается на тысячные, и с весами настоящей GPT-2 это уже заметно.

## Подсказка 2

В pre-LN нормализованный тензор живёт одну строку: он нужен только как вход подслоя. В переменную `x`
записывается сумма `x + подслой(норма(x))`, и именно этот `x` идёт во второй подслой и дальше.

## Подсказка 3

```python
if self.pre_norm:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
else:
    x = self.ln_1(x + self.attn(x))
    x = self.ln_2(x + self.mlp(x))
return x
```
