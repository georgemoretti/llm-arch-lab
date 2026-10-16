## TL;DR

GQA делит H голов запросов на H_kv групп, и каждая группа использует одну общую пару K/V.
MHA (H_kv = H) и MQA (H_kv = 1) — крайние случаи. Формула attention не меняется; меняется
только, откуда голова h берёт K и V: из KV-головы h // (H / H_kv). Выигрыш — KV-cache и его
чтение на каждом шаге генерации меньше в H / H_kv раз при почти том же качестве. Llama 3 8B:
32 головы запросов, 8 KV-голов → кеш в 4 раза меньше, 1 ГиБ на 8192 токена в bf16.

## Формула

$$\text{head}_h = \operatorname{softmax}\!\left(\frac{Q_h K_{\lfloor h/r \rfloor}^\top}{\sqrt{D}} + M\right) V_{\lfloor h/r \rfloor},\qquad r = H / H_{kv}$$

$$\text{KV-cache, байт} = 2 \cdot L \cdot H_{kv} \cdot D \cdot S \cdot B \cdot \text{bytes\_per\_elem}$$

## Формы тензоров

```
x [B,T,C] → q [B,H,T,D];  k, v [B,H_kv,T,D]  (в кеш идут они)
repeat_kv: [B,H_kv,T,D] → [B,H_kv,r,T,D] → [B,H,T,D]   (копии подряд: k0,k0,k1,k1)
attention → [B,H,T,D] → [B,T,H·D] → o_proj → [B,T,C]
```

## Что поменялось относительно GPT-2

У GPT-2 у каждой головы свои K и V (MHA) и одна проекция `c_attn` с bias. В Llama 3 — четыре
проекции без bias, а `k_proj`/`v_proj` в H / H_kv раз уже `q_proj`: головы запросов внутри группы
делят K и V. KV-cache на токен уменьшается во столько же раз.
