# Инициализация весов: Xavier и Kaiming

## Зачем это нужно

В A03 веса брались из $U(-1, 1)$, и для сети из двух слоёв это работало. В глубокой сети такая
инициализация ломает обучение ещё до первого шага. Каждый слой умножает масштаб сигнала на
некоторый множитель. Если он чуть больше 1, через 30 слоёв активации взрываются до бесконечности.
Если чуть меньше, они гаснут до нуля. В обоих случаях градиенты становятся бесполезными, и сеть не
учится.

Правильная инициализация подбирает масштаб весов так, чтобы **дисперсия сигнала сохранялась от слоя
к слою**. Две классические формулы — Xavier (Glorot, 2010) для симметричных активаций и Kaiming
(He, 2015) для ReLU. Обе есть в `torch.nn.init`, и `nn.Linear` по умолчанию тоже инициализируется
по варианту Kaiming.

## Что поменялось относительно A03

| | A03 | A06 |
|---|---|---|
| веса | $U(-1, 1)$ независимо от размера | масштаб зависит от $\text{fan\_in}$ (и $\text{fan\_out}$) |
| глубина | 2 слоя | 30 слоёв, видно взрыв и затухание |
| нелинейность | не влияет на выбор | ReLU требует множитель $\sqrt 2$ |

## Формула

Выход нейрона без bias: $y = \sum_{i=1}^{n} w_i x_i$, где $n = \text{fan\_in}$. Если веса и входы
независимы, у весов нулевое среднее, то

$$
\operatorname{Var}(y) = n \cdot \operatorname{Var}(w) \cdot \mathbb{E}[x^2].
$$

- **Линейная сеть**: чтобы $\operatorname{Var}(y) = \operatorname{Var}(x)$, нужно
  $\operatorname{Var}(w) = 1/n$.
- **ReLU** обнуляет отрицательную половину симметричного входа, поэтому
  $\mathbb{E}[\operatorname{relu}(z)^2] = \tfrac12\operatorname{Var}(z)$. Компенсация —
  $\operatorname{Var}(w) = 2/n$. Это **Kaiming**: $w \sim \mathcal N(0, \text{gain}^2/n)$ с
  $\text{gain} = \sqrt 2$.
- **Xavier** учитывает и прямой проход (нужно $1/\text{fan\_in}$), и обратный (нужно
  $1/\text{fan\_out}$) и берёт компромисс $\operatorname{Var}(w) = \text{gain}^2 \cdot \tfrac{2}{\text{fan\_in} + \text{fan\_out}}$.
  Для равномерного распределения $U(-a, a)$ дисперсия равна $a^2/3$, откуда

$$
a = \text{gain}\cdot\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}.
$$

У tanh в PyTorch $\text{gain} = 5/3$. Это эмпирическое значение: tanh немного сжимает сигнал.

**В реальных LLM** чаще всего берут простое $\mathcal N(0,\,0.02^2)$. У GPT-2 и Llama 3 8B в
`config.json` стоит `initializer_range: 0.02`
([GPT-2](https://huggingface.co/openai-community/gpt2/blob/main/config.json),
[Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)). Стабильность
там держится не на инициализации, а на нормализации перед каждым блоком и на residual-связях. GPT-2
дополнительно уменьшает начальные веса проекций, которые пишут в residual-поток, пропорционально
$1/\sqrt{N}$, где $N$ — число residual-слоёв (статья GPT-2, раздел 2.3).

## Формы тензоров

```shapes
x [n, W] — n примеров, ширина W
> W_l [W, W]: x ← act(x W_lᵀ), l = 1 … depth
x [n, W] — после каждого слоя записываем std
> std по всем элементам
stds [depth] — по одному числу на слой
axis W = ширина слоя (fan_in = fan_out = W)
```

У `nn.Linear` веса хранятся как `[fan_out, fan_in]`: строка — один выходной нейрон.

:::viz init-depth:::

## Задание

В `solution.py` (рабочая копия в `workspace/A06-init/`):

1. `gain(nonlinearity)` для `"linear"`, `"tanh"`, `"relu"`.
2. `xavier_uniform_(W, gain, generator)` и `kaiming_normal_(W, nonlinearity, generator)`. Обе заполняют `W` на
   месте внутри `torch.no_grad()` и возвращают его. Случайность бери только из `generator`
   (`W.uniform_(…, generator=…)`, `W.normal_(…, generator=…)`).
3. `activation_stds(init, act, depth, width, n, seed)` — эксперимент: прогнать $x \sim \mathcal N(0, 1)$
   через `depth` слоёв и вернуть std активаций после каждого слоя.

`torch.nn.init.*` использовать нельзя.

## Критерии готовности

- `gain`: 1, $\sqrt 2$, $5/3$;
- `xavier_uniform_` и `kaiming_normal_` выдают те же числа, что `nn.init` при том же генераторе, не
  выходят за границы и не строят граф у параметра с `requires_grad`;
- `activation_stds` совпадает с эталоном;
- на глубине 30 Kaiming + ReLU держит std порядка 1. При $\mathcal N(0, 1)$ сигнал взрывается
  (> 1e20), при $\mathcal N(0, 0.01)$ гаснет (< 1e-20). Xavier с gain = 1 гаснет с ReLU, но не на
  первом слое.

Запуск: `llmlab test A06-init`.
