import importlib.util
from pathlib import Path

import torch
from torch.testing import assert_close

from llmlab.core.gpt2 import GPT2, CausalSelfAttention, GPT2Config

C00_SOLUTION = (Path(__file__).resolve().parents[1]
                / "content/blocks/C/module-0-gpt2/C00-causal-mha/solution.py")


def _model():
    torch.manual_seed(0)
    return GPT2(GPT2Config.tiny()).eval()


def test_forward_shapes_and_loss():
    model = _model()
    cfg = model.config
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 10, cfg.vocab_size)
    # На случайной инициализации loss ≈ ln(vocab): модель ещё ничего не знает.
    assert abs(loss.item() - torch.log(torch.tensor(cfg.vocab_size)).item()) < 0.5


def test_causality():
    model = _model()
    idx = torch.randint(0, 97, (1, 12))
    idx2 = idx.clone()
    idx2[0, 8] = (idx2[0, 8] + 1) % 97
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)
    assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_weight_tying():
    model = _model()
    assert model.lm_head.weight is model.wte.weight


def test_generate_greedy_is_deterministic():
    model = _model()
    idx = torch.zeros(1, 1, dtype=torch.long)
    a = model.generate(idx, 5, temperature=0)
    b = model.generate(idx, 5, temperature=0)
    assert a.shape == (1, 6) and torch.equal(a, b)


def test_c00_reference_matches_core_attention():
    """Эталон C00 и слой из core/gpt2.py — одно и то же (те же имена параметров)."""
    spec = importlib.util.spec_from_file_location("c00_solution", C00_SOLUTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = GPT2Config.tiny()
    core = CausalSelfAttention(cfg).eval()
    c00 = mod.CausalSelfAttention(cfg.n_embd, cfg.n_head).eval()
    c00.load_state_dict(core.state_dict())
    x = torch.randn(2, 16, cfg.n_embd)
    with torch.no_grad():
        assert_close(c00(x), core(x))
