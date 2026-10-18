# Backprop руками на тензорах: Linear, ReLU, softmax + cross-entropy

## Зачем это нужно

Скалярный `Value` из A01 показывает механизм, но для настоящих сетей он слишком медленный: один узел
графа на каждое число. Фреймворки работают с **тензорами**, и у каждой операции есть backward сразу
для всего тензора. В этой задаче ты пишешь такие backward руками для трёх операций, из которых
состоит почти любой классификатор и выход любой языковой модели:

- **Linear** — матричное умножение плюс bias;
- **ReLU** — поэлементная нелинейность;
- **softmax + cross-entropy** — лосс на логитах. В языковой модели это лосс предсказания
  следующего токена по словарю.

Здесь же два практических приёма, которые встречаются в каждой реализации. Первый — **logsumexp-трюк**:
без него `exp` переполняется на больших логитах. Второй — **градиент $p - y$**: формула, которую
стоит помнить наизусть.

## Что поменялось относительно A03

| | A03: MLP на Value | A04: тензоры |
|---|---|---|
| единица графа | одно число | целый тензор |
| backward | autograd обходит граф скаляров | ты пишешь backward каждого слоя |
| что хранится между проходами | весь граф | только `cache` каждого слоя |
| лосс | MSE | softmax + cross-entropy по классам |
| батч | цикл по примерам | одна матрица `[N, D]` |

## Формула

**Linear.** $y = x W^\top + b$, где $x \in \mathbb{R}^{N \times D_{in}}$, $W \in \mathbb{R}^{D_{out} \times D_{in}}$.
Градиенты при известном $dy = \partial L / \partial y$:

$$
dx = dy\,W, \qquad dW = dy^\top x, \qquad db = \sum_{n=1}^{N} dy_n.
$$

Проще всего проверять такие формулы **по формам**: у $dW$ форма $W$. Из $dy$ формы $[N, D_{out}]$
и $x$ формы $[N, D_{in}]$ матрицу $[D_{out}, D_{in}]$ даёт только $dy^\top x$.

**ReLU.** $dx = dy \odot \mathbb{1}[x > 0]$.

**Softmax + cross-entropy.** Для логитов $z \in \mathbb{R}^{V}$ и правильного класса $y$:

$$
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}, \qquad
\ell = -\log p_y = \log\sum_k e^{z_k} - z_y, \qquad
\frac{\partial \ell}{\partial z_j} = p_j - \mathbb{1}[j = y].
$$

Градиент такой простой, потому что log и exp сокращаются. Поэтому softmax и cross-entropy всегда
считают вместе, по логитам. Для среднего по батчу каждая строка градиента ещё делится на $N$.

**Logsumexp-трюк.** $e^{1000}$ — это бесконечность даже во float64 (максимум около $e^{709}$). Но
softmax не меняется от сдвига всех логитов на константу, поэтому вычитаем максимум:

$$
\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}, \qquad m = \max_k z_k.
$$

После сдвига самый большой показатель равен 0: переполнения нет, и хотя бы одно слагаемое равно 1,
поэтому логарифм от нуля тоже не возникает.

## Формы тензоров

```shapes
x [N, D] — батч входов
> Linear 1: x W1ᵀ + b1, W1 [H, D]
h [N, H]
> ReLU
a [N, H]
> Linear 2: a W2ᵀ + b2, W2 [V, H]
logits [N, V]
> − max по строке, log-softmax, −mean log p[i, y_i]
loss [] — средний по батчу
## backward
dlogits [N, V] — (p − onehot) / N
> Linear 2 backward: dW2 = dlogitsᵀ a, db2 = Σ по N
da [N, H]
> ReLU backward: маска a > 0
dh [N, H]
> Linear 1 backward
dW1 [H, D] — форма W1
axis V = число классов (в LM — размер словаря)
```

:::viz softmax-ce:::

## Задание

В `solution.py` (рабочая копия в `workspace/A04-tensor-backprop/`) реализуй:

1. `linear_forward` / `linear_backward`, `relu_forward` / `relu_backward`. `forward`
   возвращает `(выход, cache)`, `backward` получает `cache` своего `forward`.
2. `softmax_cross_entropy(logits, targets)` → `(loss, dlogits)` со сдвигом на максимум.
   `loss` — скаляр-тензор, среднее по батчу.
3. `mlp_loss_and_grads(x, targets, params)` — двухслойная сеть
   Linear → ReLU → Linear → CE и её градиенты в словаре с теми же ключами.

Нельзя: `.backward()`, `torch.autograd.*`, `softmax`, `log_softmax`, `logsumexp`,
`cross_entropy`, `nll_loss`. Можно: `@`, `exp`, `log`, `sum`, `max`, индексацию.

## Критерии готовности

- `linear_forward` даёт $x W^\top + b$ формы `[N, D_out]`;
- `dx`, `dW`, `db` совпадают с autograd и имеют формы `x`, `W`, `b`;
- `relu_backward` пропускает градиент только там, где вход > 0;
- `loss` равен `F.cross_entropy` (среднему), `dlogits` = $(p - \text{onehot})/N$, и сумма каждой
  строки `dlogits` равна 0;
- логиты около 1000–2000 дают конечные и точные `loss` и `dlogits`;
- градиенты всех четырёх параметров двухслойной сети совпадают с autograd.

Запуск: `llmlab test A04-tensor-backprop`.
