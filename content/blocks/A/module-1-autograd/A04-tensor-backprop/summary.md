## TL;DR

У каждого слоя есть forward, который возвращает `(выход, cache)`, и backward, который по градиенту
выхода и `cache` даёт градиенты входа и параметров. Linear: $dx = dy W$, $dW = dy^\top x$,
$db = \sum dy$ (проверяется по формам). Softmax + CE по логитам даёт градиент $p - \text{onehot}$
(для среднего по батчу делённый на N). Перед `exp` вычитают максимум строки (logsumexp-трюк),
иначе логиты около 1000 дают inf/nan.

## Формула

$$\ell = \log\sum_k e^{z_k} - z_y,\quad \frac{\partial \ell}{\partial z} = \operatorname{softmax}(z) - e_y,\quad \log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}$$

## Формы тензоров

```
x [N,D] → Linear → [N,H] → ReLU → [N,H] → Linear → logits [N,V] → CE → loss []
backward: dlogits [N,V] = (p − onehot)/N → dW2 [V,H], da [N,H] → маска → dW1 [H,D]
```

## Что поменялось относительно A03

Вместо графа скаляров — тензоры и написанный руками backward каждого слоя. Вместо MSE — softmax +
cross-entropy по классам с численно устойчивым log-softmax. Так устроен выход любой языковой модели:
логиты по словарю и cross-entropy со следующим токеном.
