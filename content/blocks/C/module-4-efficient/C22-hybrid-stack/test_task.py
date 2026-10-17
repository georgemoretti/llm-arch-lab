"""Тесты «Гибридный стек»: гибридный стек 3 : 1 и память кеша.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: расписание слоёв как в Qwen3-Next ((i + 1) % 4 == 0 → full); блоки — Pre-Norm с любым mixer
(у Gated DeltaNet выход — кортеж); стек не утекает в будущее; память: KV-cache full-слоёв растёт с длиной,
состояния linear-слоёв — нет; у Qwen3-Next гибрид выгоднее «всё full» после 512 токенов и стремится к ¼.
"""

import pytest
import torch
from torch.testing import assert_close

from llmlab.core.efficient import GatedAttention, GatedDeltaNet

B, T, C, D_FF = 2, 10, 32, 48
CFG = dict(d_model=C, n_layers=8, full_attention_interval=4, d_ff=D_FF, n_heads=4, n_kv_heads=2, head_dim=8,
           linear_heads=2, linear_k_dim=8, linear_v_dim=8)
# Qwen/Qwen3-Next-80B-A3B-Instruct config.json
QWEN = dict(n_layers=48, interval=4, n_kv_heads=2, head_dim=256, linear_heads=32, linear_k_dim=128, linear_v_dim=128)


def _x(t=T, seed=1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _stack(impl, **over):
    torch.manual_seed(0)
    return impl.HybridStack(**{**CFG, **over}).eval()


# ---------------------------------------------------------------------------
# Расписание слоёв
# ---------------------------------------------------------------------------

def test_layer_types_qwen3_next(impl):
    """hybrid_layer_types: 48 слоёв, interval 4 → full в слоях 3, 7, …, 47 (12 шт.), остальные linear"""
    types = impl.hybrid_layer_types(QWEN["n_layers"], QWEN["interval"])
    assert len(types) == 48, f"Нужен список из 48 типов, получено {len(types)}."
    assert set(types) <= {"full_attention", "linear_attention"}, (
        f"Типы слоёв — строки 'full_attention' / 'linear_attention' (как layer_types в HF), получено {set(types)}.")
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    if full != list(range(3, 48, 4)):
        hint = " Похоже на i % interval == 0: тогда full — первый в группе (0, 4, …), а у Qwen3-Next — последний." \
            if full[:2] == [0, 4] else ""
        pytest.fail(f"Full-слои: {full[:6]}…, ожидалось [3, 7, 11, …, 47]: правило (i + 1) % interval == 0.{hint}",
                    pytrace=False)
    assert impl.hybrid_layer_types(6, 1) == ["full_attention"] * 6, "interval = 1 → все слои full (обычный трансформер)."


def test_stack_mixers(impl):
    """HybridStack: mixer слоя — GatedAttention для full и GatedDeltaNet для linear"""
    m = _stack(impl)
    kinds = ["full" if isinstance(l.mixer, GatedAttention) else "linear" if isinstance(l.mixer, GatedDeltaNet) else "?"
             for l in m.layers]
    assert kinds == ["linear", "linear", "linear", "full"] * 2, f"Mixer-ы по слоям: {kinds}."


# ---------------------------------------------------------------------------
# Блок и стек
# ---------------------------------------------------------------------------

def _block_oracle(block, x):
    h = block.mixer(block.input_layernorm(x))
    h = h[0] if isinstance(h, tuple) else h
    x = x + h
    return x + block.mlp(block.post_attention_layernorm(x))


def test_block_forward(impl):
    """HybridBlock: Pre-Norm residual с любым mixer — x + mixer(norm(x)), затем x + mlp(norm(x))"""
    m = _stack(impl)
    x = _x()
    for idx in (0, 3):                          # linear и full
        block = m.layers[idx]
        with torch.no_grad():
            out = block(x)
            expected = _block_oracle(block, x)
        assert torch.is_tensor(out), (
            f"Блок {idx} вернул {type(out).__name__}: GatedDeltaNet возвращает (y, state) — в residual идёт только y.")
        assert_close(out, expected, atol=1e-5, rtol=1e-4, msg=lambda _: (
            f"Блок {idx} ({m.layer_types[idx]}) не совпадает с Pre-Norm формулой: h = x + mixer(input_layernorm(x)), "
            "out = h + mlp(post_attention_layernorm(h))."))


def test_stack_forward(impl):
    """HybridStack.forward: блоки по порядку, в конце финальная RMSNorm"""
    m = _stack(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
        h = x
        for block in m.layers:
            h = _block_oracle(block, h)
        expected = m.norm(h)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    assert_close(out, expected, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Стек не совпадает с последовательным применением блоков + финальная norm."))


def test_stack_causal(impl):
    """HybridStack: выход в позиции t не зависит от токенов t + 1, … (ни в attention-, ни в DeltaNet-слоях)"""
    m = _stack(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
        x2 = x.clone()
        x2[:, 6:] = torch.randn(B, T - 6, C)
        out2 = m(x2)
    assert_close(out[:, :6], out2[:, :6], atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Выход позиций < 6 изменился после замены токенов 6… → будущее протекает в одном из слоёв."))


# ---------------------------------------------------------------------------
# Память
# ---------------------------------------------------------------------------

def _bytes(impl, types, seq):
    return impl.hybrid_cache_bytes(types, seq, QWEN["n_kv_heads"], QWEN["head_dim"], QWEN["linear_heads"],
                                   QWEN["linear_k_dim"], QWEN["linear_v_dim"])


def test_cache_bytes_formula(impl):
    """hybrid_cache_bytes: full-слой — 2·H_kv·d·S, linear-слой — H·dk·dv, умножить на batch и байты"""
    types = ["linear_attention", "full_attention", "linear_attention"]
    got = impl.hybrid_cache_bytes(types, 100, 2, 8, 3, 4, 5, batch=2, bytes_per_elem=4)
    expected = (1 * 2 * 2 * 8 * 100 + 2 * 3 * 4 * 5) * 2 * 4
    if got == (3 * 2 * 2 * 8 * 100) * 2 * 4:
        pytest.fail("Все слои посчитаны как full: у linear-слоя вместо KV-cache — состояние H·dk·dv.", pytrace=False)
    assert got == expected, (
        f"Получено {got} Б, ожидалось {expected}: 1 full-слой × 2·H_kv·d·S + 2 linear-слоя × H·dk·dv, × batch × байты.")


def test_cache_qwen3_next(impl):
    """Qwen3-Next: состояния не растут с длиной; гибрид выгоднее «всё full» после 512 токенов; на 256K — ≈ ¼"""
    hybrid = impl.hybrid_layer_types(48, 4)
    full = impl.hybrid_layer_types(48, 1)
    s1, s2 = _bytes(impl, hybrid, 1000), _bytes(impl, hybrid, 2000)
    assert s2 - s1 == 12 * 2 * 2 * 256 * 1000 * 2, (
        "При росте контекста на 1000 токенов растёт только KV-cache 12 full-слоёв: 12·2·2·256·1000·2 Б.")
    assert _bytes(impl, hybrid, 512) == _bytes(impl, full, 512), (
        "На 512 токенах гибрид и «всё full» должны занимать одинаково: состояние Gated DeltaNet (32·128·128) = "
        "KV-cache одного full-слоя (2·2·256 на токен) на 512 токенов.")
    assert _bytes(impl, hybrid, 100) > _bytes(impl, full, 100), "На коротком контексте состояния дороже KV-cache."
    ratio = _bytes(impl, hybrid, 262144) / _bytes(impl, full, 262144)
    assert 0.25 < ratio < 0.26, (
        f"На 262 144 токенах гибрид / всё full = {ratio:.3f}, ожидалось ≈ 0.25 (KV-cache только у каждого 4-го слоя).")
