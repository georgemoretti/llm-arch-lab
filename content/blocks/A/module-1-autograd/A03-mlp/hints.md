## Подсказка 1

Нейрон: начни с `act = self.b` и прибавляй `wi * xi` для каждой пары `zip(self.w, x)`. `x` может
быть списком чисел или списком `Value`: благодаря `_wrap` в `Value` оба случая работают. В конце
применяй `act.tanh()`, только если `self.nonlin`.

## Подсказка 2

`parameters()` у слоя и сети — плоский список, собранный из параметров частей:
`[p for n in self.neurons for p in n.parameters()]`. `MLP.__call__`: `for layer in self.layers: x = layer(x)`.

## Подсказка 3

Цикл обучения на одном шаге:

```python
preds = [model(x)[0] for x in xs]
loss = mse_loss(preds, ys)
zero_grad(model.parameters())   # backward ПРИБАВЛЯЕТ к .grad
loss.backward()
sgd_step(model.parameters(), lr)
```
