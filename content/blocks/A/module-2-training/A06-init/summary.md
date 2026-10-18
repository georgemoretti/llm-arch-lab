## TL;DR

Каждый слой умножает дисперсию сигнала на $\text{fan\_in}\cdot\operatorname{Var}(w)\cdot(\text{множитель активации})$.
Чтобы через десятки слоёв сигнал не взорвался и не погас, этот множитель должен быть ≈ 1. Kaiming:
$\mathcal N(0, 2/\text{fan\_in})$ для ReLU. Xavier: $\operatorname{Var} = 2/(\text{fan\_in} + \text{fan\_out})$
для симметричных активаций. В LLM обычно $\mathcal N(0, 0.02^2)$, а устойчивость обеспечивают
нормализация и residual.

## Формула

$$\operatorname{Var}(y) = \text{fan\_in}\,\operatorname{Var}(w)\,\mathbb{E}[x^2];\quad \text{Kaiming: } \operatorname{Var}(w) = \tfrac{\text{gain}^2}{\text{fan\_in}};\quad \text{Xavier: } a = \text{gain}\sqrt{\tfrac{6}{\text{fan\_in}+\text{fan\_out}}}$$

## Формы тензоров

```
W [fan_out, fan_in]; x [n, W] → act(x Wᵀ) × depth → stds [depth]
```

## Что поменялось относительно A03

Вместо $U(-1, 1)$ для любого слоя масштаб весов подбирается по размеру слоя и типу нелинейности.
Глубокая сеть стартует с сигналом порядка 1 на каждом слое.
