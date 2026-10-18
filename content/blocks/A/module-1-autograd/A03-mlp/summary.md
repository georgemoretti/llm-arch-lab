## TL;DR

MLP — цепочка слоёв из нейронов $\tanh(b + w\cdot x)$, выходной слой линейный. Цикл обучения:
forward → лосс → `zero_grad` → `backward` → шаг SGD $\theta \leftarrow \theta - \eta\nabla\theta$.
Градиенты нужно обнулять, потому что `backward` их накапливает. Без нелинейности сеть линейна и не
разделит кольца.

## Формула

$$y = \tanh\big(b + \textstyle\sum_i w_i x_i\big),\qquad L = \tfrac{1}{N}\textstyle\sum_k (\hat y_k - y_k)^2,\qquad \theta \leftarrow \theta - \eta\,\partial L/\partial\theta$$

## Формы тензоров

```
x [2] → Layer(2→8, tanh) → [8] → Layer(8→1, линейный) → ŷ [1];  MSE по батчу → L [] → backward → 33 градиента
```

## Что поменялось относительно A01

Граф строится не для одного выражения, а для сети и лосса по батчу. Градиенты используются для
шага оптимизатора. Появились `parameters()`, `zero_grad` и цикл обучения: та же схема, что у
`torch.optim` в обучении любой модели.
