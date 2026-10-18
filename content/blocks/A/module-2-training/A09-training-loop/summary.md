## TL;DR

Шаг обучения: lr по расписанию → батч → лосс → `zero_grad` → `backward` → clip → `step`. Валидация —
в `eval()` и под `no_grad`, затем возврат в прежний режим. Первая проверка нового цикла —
переобучиться на одном батче: если лосс не идёт к нулю, сломан шаг, а не модель.

## Формула

$$g = \nabla_\theta\,\ell(f_\theta(x_t), y_t);\ g \leftarrow \operatorname{clip}(g);\ \theta \leftarrow \operatorname{step}(\theta, g, \eta(t))$$

## Формы тензоров

```
X [N,D] → get_batch → xb [B,D] → model → logits [B,K] → loss [];  валидация: X_val [M,D] → val_loss []
```

## Что поменялось относительно A03

Вместо полного батча на скалярах — `nn.Module`, случайные батчи, расписание lr, clipping, dropout и
валидация в режиме eval. Добавился smoke-тест переобучением как первая проверка цикла.
