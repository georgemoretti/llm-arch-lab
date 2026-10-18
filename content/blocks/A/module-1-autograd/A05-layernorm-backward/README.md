# Backward LayerNorm руками

## Зачем это нужно

LayerNorm стоит в каждом блоке GPT-2, а RMSNorm (C01) — в каждом блоке Llama. Их backward — частый
вопрос на собеседованиях, потому что в нём легко ошибиться: среднее $\mu$ и дисперсия $\sigma^2$
**сами зависят от $x$**. Если считать их константами, получится формула, которая выглядит правдоподобно,
но даёт неверный градиент. Модель с такой ошибкой обучается заметно хуже, и найти причину трудно.

После этой задачи ты сможешь вывести backward любой нормализации. Тот же приём работает для
RMSNorm (там один путь вместо двух) и для softmax.

## Что поменялось относительно A04

| | A04 | A05 |
|---|---|---|
| операции | поэлементные и матричные | статистики по строке: $\mu$, $\sigma$ |
| якобиан | диагональный (ReLU) или $W$ (Linear) | плотный $C \times C$ внутри каждой строки |
| ловушка | формы, деление на N | пути через $\mu$ и $\sigma$ |
| свойство для проверки | $\sum_j \partial\ell/\partial z_j = 0$ у softmax-CE | $\sum_j dx_j = 0$ у каждой строки |

## Формула

Прямой проход по последней оси $C$ (для одной строки $x \in \mathbb{R}^C$):

$$
\mu = \frac{1}{C}\sum_i x_i,\quad
\sigma^2 = \frac{1}{C}\sum_i (x_i - \mu)^2,\quad
\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}},\quad
y = \hat x \odot \gamma + \beta.
$$

Дисперсия **смещённая**: делим на $C$, а не на $C - 1$.

Обратный проход. С параметрами просто: $d\beta = \sum dy$ и $d\gamma = \sum dy \odot \hat x$, где сумма
берётся по всем осям, кроме $C$. Для $dx$ обозначим $g = dy \odot \gamma$ (это $\partial L / \partial \hat x$) и
$r = 1/\sqrt{\sigma^2 + \varepsilon}$. Каждый $\hat x_i$ зависит от $x_j$ тремя путями: напрямую, через
$\mu$ и через $\sigma^2$. Дифференцируем:

$$
\frac{\partial \hat x_i}{\partial x_j} = r\Big(\delta_{ij} - \frac{1}{C} - \frac{\hat x_i\,\hat x_j}{C}\Big),
$$

а затем сворачиваем с $g$:

$$
dx = r\,\Big(g \;-\; \operatorname{mean}(g) \;-\; \hat x \odot \operatorname{mean}(g \odot \hat x)\Big).
$$

Смысл формулы: из $g$ вычитаются его проекция на вектор из единиц и (при $\varepsilon = 0$) проекция на
$\hat x$. LayerNorm не замечает сдвига $x + c\,\mathbf 1$ и растяжения $x \to kx$, поэтому градиент в
этих направлениях обязан быть нулевым. Отсюда два свойства, которые проверяют тесты:
$\sum_j dx_j = 0$ и $dx(kx) = dx(x)/k$.

## Формы тензоров

```shapes
x [B, T, C]
> μ, σ² по оси C (keepdim)
μ, rstd [B, T, 1] — одна пара чисел на токен
> x̂ = (x − μ)·rstd; y = x̂·γ + β
y [B, T, C]
## backward
dy [B, T, C]
fork
> Σ по B, T от dy·x̂
dgamma [C]
branch
> Σ по B, T от dy
dbeta [C]
branch
> g = dy·γ; dx = rstd·(g − mean_C g − x̂·mean_C(g·x̂))
dx [B, T, C]
join
```

:::viz layernorm-grad:::

## Задание

В `solution.py` (рабочая копия в `workspace/A05-layernorm-backward/`) реализуй:

1. `layernorm_forward(x, gamma, beta, eps)` → `(y, cache)` для `x` формы `[..., C]`.
2. `layernorm_backward(dy, cache)` → `(dx, dgamma, dbeta)`.

Нельзя: `F.layer_norm`, `nn.LayerNorm`, `.backward()`, `torch.autograd`. Выведи формулу на бумаге,
прежде чем писать код. Это и есть суть задачи.

## Критерии готовности

- forward совпадает с `F.layer_norm`, а дисперсия смещённая;
- `dx`, `dgamma`, `dbeta` совпадают с autograd на `[B, T, C]`, у `dgamma` и `dbeta` форма `[C]`;
- сумма `dx` по оси C равна нулю в каждой строке;
- для входа `10·x` (при $\varepsilon = 0$) `dx` ровно в 10 раз меньше, а `dgamma` тот же;
- всё верно при $C = 2$ и при большом $\varepsilon = 0.5$.

Запуск: `llmlab test A05-layernorm-backward`.
