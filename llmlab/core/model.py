"""Конфигурируемая модель (Фаза 8): одна LLM, собранная из YAML через реестр компонентов.

Все задачи блока C реализовали по одной детали. Здесь они собираются в одну модель, где каждая
деталь выбирается конфигом:

    norm        layernorm | rmsnorm | gemma_rmsnorm (вес 1 + w)          — GPT-2, C01, C08
    placement   pre | post | olmo | sandwich                            — C08
    position    learned | rope | none (+ расписание NoPE, θ для local)  — GPT-2, C03, C10
    attention   gqa (MHA/GQA/MQA, QK-Norm, окно, гейт, partial RoPE) | mla — C04, C07, C09, C19, C16b
    linear      gated_deltanet (гибрид 3 : 1)                            — C21, C22
    ffn         gelu | swiglu | geglu                                    — GPT-2, C02
    moe         роутер + эксперты + shared expert                        — C12–C15

Расписания слоёв берутся из эталонов: ``layer_types`` (Gemma, C07), ``rope_layers`` (SmolLM3, C10),
``mlp_layer_types`` (DeepSeek, C15), ``hybrid_layer_types`` (Qwen3-Next, C22).

Почему attention здесь своя, а не ``LlamaAttention``/``QKNormAttention``/``GatedAttention`` из эталонов:
каждый эталон показывает ОДНУ идею, а реальной модели нужно их сочетание (Qwen3 = GQA + QK-Norm +
head_dim ≠ d_model / H + KV-cache; Gemma 3 = то же + окно + свой θ + (1 + w)-нормы). Поэтому
``Attention`` собрана из функций эталонов (``rope_cos_sin``, ``apply_partial_rope``, ``repeat_kv``,
``KVCache``, ``RingKVCache``), а то, что её частные случаи совпадают с каждым эталоном на одних
весах, проверяет ``tests/test_core_model.py``. MLA, Gated DeltaNet, MoE и FFN — классы эталонов как есть.

Имена параметров — как в Hugging Face ``transformers`` без префикса ``model.`` (``embed_tokens``,
``layers.{i}.self_attn.q_proj``, ``layers.{i}.mlp.gate_proj``, ``norm``, ``lm_head``): загрузка весов
(``llmlab/core/hf.py``) — это в основном снятие префикса.

Готовые конфиги — ``llmlab/configs/*.yaml``; ``Model.from_config("qwen3-0.6b")``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from llmlab.core.efficient import GatedDeltaNet, apply_partial_rope, hybrid_layer_types
from llmlab.core.gemma_olmo import RingKVCache, layer_types, rope_layers
from llmlab.core.gpt2 import MLP as GPT2MLP
from llmlab.core.gpt2 import GPT2Config
from llmlab.core.llama import KVCache, RMSNorm, SwiGLUFFN, repeat_kv, rms_norm, rope_cos_sin
from llmlab.core.mla import LatentCache, MLARopeAttention
from llmlab.core.moe import DeepSeekMoE, mlp_layer_types

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# ============================================================================ конфиг


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")   # опечатка в YAML → понятная ошибка, а не тихий дефолт


class NormConfig(_Section):
    type: Literal["layernorm", "rmsnorm", "gemma_rmsnorm"] = "rmsnorm"
    eps: float = 1e-6
    placement: Literal["pre", "post", "olmo", "sandwich"] = "pre"


class PositionConfig(_Section):
    type: Literal["learned", "rope", "none"] = "rope"
    theta: float = 10000.0
    local_theta: float | None = None             # Gemma 3: свой θ у sliding-слоёв
    partial_rotary_factor: float = 1.0           # Qwen3-Next: RoPE на четверти канала головы
    no_rope_layer_interval: int | None = None    # SmolLM3: каждый k-й слой без RoPE
    max_positions: int = 2048                    # только для learned (GPT-2: 1024)


class AttentionConfig(_Section):
    type: Literal["gqa", "mla"] = "gqa"
    n_heads: int
    n_kv_heads: int | None = None                # None → n_heads (MHA)
    head_dim: int | None = None                  # None → d_model // n_heads
    bias: bool = False                           # GPT-2: bias у q/k/v/o
    qk_norm: Literal["head", "full"] | None = None
    output_gate: bool = False                    # Qwen3-Next: σ(gate) ⊙ attn перед o_proj
    query_scale: float | None = None             # множитель логитов; None → 1/√head_dim
    window: int | None = None                    # окно sliding-слоёв
    sliding_window_pattern: int | None = None    # Gemma: каждый k-й слой full; None + window → все sliding
    # MLA (C16b)
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None


class LinearAttentionConfig(_Section):
    type: Literal["gated_deltanet"] = "gated_deltanet"
    full_attention_interval: int = 4             # Qwen3-Next: 3 linear : 1 full
    n_heads: int
    head_k_dim: int
    head_v_dim: int


class FFNConfig(_Section):
    type: Literal["gelu", "swiglu", "geglu"] = "swiglu"
    d_hidden: int
    bias: bool = False


class MoEConfig(_Section):
    n_routed: int
    n_shared: int = 0
    top_k: int
    d_expert: int
    scoring_func: Literal["softmax", "sigmoid"] = "softmax"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1


class ModelConfig(_Section):
    name: str = "custom"
    source: str | None = None                    # откуда взяты числа (config.json / статья)
    vocab_size: int
    d_model: int
    n_layers: int
    tie_embeddings: bool = True
    embed_scale: bool = False                    # Gemma: эмбеддинги × √d_model
    norm: NormConfig = NormConfig()
    position: PositionConfig = PositionConfig()
    attention: AttentionConfig
    linear_attention: LinearAttentionConfig | None = None
    ffn: FFNConfig
    moe: MoEConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> "ModelConfig":
        a = self.attention
        if a.type == "gqa":
            H, H_kv = a.n_heads, a.n_kv_heads or a.n_heads
            if H % H_kv:
                raise ValueError(f"n_heads = {H} не делится на n_kv_heads = {H_kv}: GQA делит головы на группы поровну")
            if a.head_dim is None and self.d_model % H:
                raise ValueError(f"d_model = {self.d_model} не делится на n_heads = {H}; задай head_dim явно")
        else:
            missing = [f for f in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
                       if getattr(a, f) is None]
            if missing:
                raise ValueError(f"attention.type = mla требует {', '.join(missing)}")
        if self.position.type == "learned" and self.position.no_rope_layer_interval:
            raise ValueError("no_rope_layer_interval имеет смысл только при position.type = rope")
        return self

    @property
    def head_dim(self) -> int:
        return self.attention.head_dim or self.d_model // self.attention.n_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def named(cls, name: str) -> "ModelConfig":
        """Готовый конфиг из ``llmlab/configs/<name>.yaml``."""
        path = CONFIGS_DIR / f"{name}.yaml"
        if not path.exists():
            known = ", ".join(sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml")))
            raise FileNotFoundError(f"Нет конфига {name!r}. Есть: {known}")
        return cls.from_yaml(path)


def available_configs() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))


@dataclass(frozen=True)
class LayerSpec:
    """Что стоит в слое i: какой mixer, окно, есть ли RoPE и с каким θ, dense или MoE FFN."""

    mixer: str                     # gqa | mla | gated_deltanet
    window: int | None
    use_rope: bool
    rope_theta: float
    ffn: str                       # dense | moe


def layer_specs(cfg: ModelConfig) -> list[LayerSpec]:
    """Расписание слоёв из конфига — через функции-расписания эталонов C07, C10, C15, C22."""
    n, a, p = cfg.n_layers, cfg.attention, cfg.position
    if cfg.linear_attention is not None:
        mixers = [a.type if t == "full_attention" else cfg.linear_attention.type
                  for t in hybrid_layer_types(n, cfg.linear_attention.full_attention_interval)]
    else:
        mixers = [a.type] * n
    if a.window is None:
        windows = [None] * n
    elif a.sliding_window_pattern:
        windows = [a.window if t == "sliding_attention" else None
                   for t in layer_types(n, a.sliding_window_pattern)]
    else:
        windows = [a.window] * n                                       # Mistral: все слои с окном
    if p.type != "rope":
        ropes = [False] * n
    elif p.no_rope_layer_interval:
        ropes = rope_layers(n, p.no_rope_layer_interval)
    else:
        ropes = [True] * n
    ffns = (mlp_layer_types(n, cfg.moe.first_k_dense_replace, cfg.moe.moe_layer_freq)
            if cfg.moe is not None else ["dense"] * n)
    return [LayerSpec(m, w, r, p.local_theta if (w is not None and p.local_theta) else p.theta, f)
            for m, w, r, f in zip(mixers, windows, ropes, ffns)]


# ============================================================================ реестр

REGISTRY: dict[str, dict[str, Callable]] = {"norm": {}, "mixer": {}, "ffn": {}}


def register(kind: str, name: str):
    """Декоратор: ``@register("ffn", "geglu")`` добавляет фабрику компонента в реестр."""
    def deco(fn: Callable) -> Callable:
        REGISTRY[kind][name] = fn
        return fn
    return deco


def build(kind: str, name: str, *args, **kwargs) -> nn.Module:
    try:
        factory = REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"Компонента {kind}={name!r} нет в реестре. Есть: {sorted(REGISTRY[kind])}") from None
    return factory(*args, **kwargs)


# ---------------------------------------------------------------------------- нормы


class GemmaRMSNorm(RMSNorm):
    """RMSNorm с весом (1 + w), w инициализирован нулями (Gemma 2/3, Qwen3-Next). При загрузке весов
    это важно: в чекпойнте лежит w, а не 1 + w."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)
        nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Как в HF Gemma3RMSNorm: (1 + w) умножается в fp32, потом приведение типа
        return (rms_norm(x.float(), 1.0 + self.weight.float(), self.eps)).to(x.dtype)


register("norm", "rmsnorm")(lambda dim, eps: RMSNorm(dim, eps))
register("norm", "gemma_rmsnorm")(lambda dim, eps: GemmaRMSNorm(dim, eps))
register("norm", "layernorm")(lambda dim, eps: nn.LayerNorm(dim, eps=eps))


# ---------------------------------------------------------------------------- mixers


class Attention(nn.Module):
    """Causal attention со всеми «ручками» блока C, собранная из функций эталонов.

    GQA (C04) · RoPE / partial RoPE / NoPE (C03, C19, C10) · QK-Norm по голове или по проекции (C09) ·
    окно (C07) с кольцевым кешем (C11) · KV-cache (C05) · выходной гейт (C19) · bias (GPT-2).
    x [B, T, C] — новые токены, cache — KVCache / RingKVCache прошлых (или None) → [B, T, C].
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, *, bias: bool = False,
                 qk_norm: str | None = None, norm: Callable[[int], nn.Module] | None = None,
                 use_rope: bool = True, rope_theta: float = 10000.0, partial_rotary_factor: float = 1.0,
                 window: int | None = None, output_gate: bool = False, query_scale: float | None = None):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.qk_norm, self.use_rope, self.rope_theta = qk_norm, use_rope, rope_theta
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.window, self.output_gate = window, output_gate
        self.scale = query_scale if query_scale is not None else head_dim ** -0.5
        q_out = n_heads * head_dim * (2 if output_gate else 1)          # [запрос | гейт] на голову, как в HF
        self.q_proj = nn.Linear(d_model, q_out, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)
        if qk_norm is not None:
            norm = norm or (lambda dim: RMSNorm(dim, 1e-6))
            full = qk_norm == "full"
            self.q_norm = norm(n_heads * head_dim if full else head_dim)
            self.k_norm = norm(n_kv_heads * head_dim if full else head_dim)

    def new_cache(self):
        return RingKVCache(self.window) if self.window is not None else KVCache()

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        start = cache.seq_len if cache is not None else 0
        q_pos = torch.arange(start, start + T, device=x.device)

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        gate = None
        if self.output_gate:
            q, gate = q.view(B, T, H, 2 * D).chunk(2, dim=-1)
            gate = gate.reshape(B, T, H * D)
        if self.qk_norm == "full":                                       # OLMo 2: по всей проекции
            q, k = self.q_norm(q.reshape(B, T, H * D)), self.k_norm(k)
        q, k = q.reshape(B, T, H, D), k.view(B, T, H_kv, D)
        if self.qk_norm == "head":                                       # Qwen3, Gemma 3: по голове
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)                      # [B, H, T, D], [B, H_kv, T, D]
        v = v.view(B, T, H_kv, D).transpose(1, 2)
        if self.use_rope:                                                # норма — ДО RoPE (C09)
            cos, sin = rope_cos_sin(q_pos, self.rotary_dim, self.rope_theta)
            q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)

        if isinstance(cache, RingKVCache):
            k, v, k_pos = cache.update(k, v)
        elif cache is not None:
            k, v = cache.update(k, v)
            k_pos = torch.arange(k.shape[2], device=x.device)
        else:
            k_pos = q_pos
        k, v = repeat_kv(k, H // H_kv), repeat_kv(v, H // H_kv)

        dist = q_pos[:, None] - k_pos[None, :]                           # [T, S]
        allowed = (k_pos[None, :] >= 0) & (dist >= 0)
        if self.window is not None:
            allowed = allowed & (dist < self.window)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        out = F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, H * D)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)


class MLAMixer(MLARopeAttention):
    """MLA из C16b + интерфейс mixer-а (латентный кеш)."""

    def new_cache(self):
        return LatentCache()


class _State:
    """Кеш linear-слоя: одно состояние фиксированного размера вместо растущего KV."""

    def __init__(self):
        self.state: torch.Tensor | None = None
        self.seq_len = 0


class DeltaNetMixer(GatedDeltaNet):
    """Gated DeltaNet из C21 + интерфейс mixer-а: кеш — состояние S [B, H, dk, dv]."""

    def new_cache(self):
        return _State()

    def forward(self, x: torch.Tensor, cache: _State | None = None) -> torch.Tensor:
        y, state = super().forward(x, cache.state if cache is not None else None)
        if cache is not None:
            cache.state = state
            cache.seq_len += x.shape[1]
        return y


@register("mixer", "gqa")
def _attention(cfg: ModelConfig, spec: LayerSpec) -> nn.Module:
    a = cfg.attention
    return Attention(cfg.d_model, a.n_heads, a.n_kv_heads or a.n_heads, cfg.head_dim, bias=a.bias,
                     qk_norm=a.qk_norm, norm=lambda dim: build("norm", cfg.norm.type, dim, cfg.norm.eps),
                     use_rope=spec.use_rope, rope_theta=spec.rope_theta,
                     partial_rotary_factor=cfg.position.partial_rotary_factor, window=spec.window,
                     output_gate=a.output_gate, query_scale=a.query_scale)


@register("mixer", "mla")
def _mla(cfg: ModelConfig, spec: LayerSpec) -> nn.Module:
    a = cfg.attention
    return MLAMixer(cfg.d_model, a.n_heads, a.kv_lora_rank, a.qk_nope_head_dim, a.qk_rope_head_dim,
                    a.v_head_dim, spec.rope_theta)


@register("mixer", "gated_deltanet")
def _deltanet(cfg: ModelConfig, spec: LayerSpec) -> nn.Module:
    la = cfg.linear_attention
    return DeltaNetMixer(cfg.d_model, la.n_heads, la.head_k_dim, la.head_v_dim)


# ---------------------------------------------------------------------------- FFN


class GeGLUFFN(SwiGLUFFN):
    """Gemma: тот же гейтированный FFN, что SwiGLU (C02), но гейт — GELU (tanh-аппроксимация)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class MoEFFN(DeepSeekMoE):
    """MoE из C15 с интерфейсом FFN: возвращает только выход, логиты роутера — в ``last_router_logits``
    (нужны для aux loss C14 при обучении)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, self.last_router_logits = super().forward(x)
        return out


@register("ffn", "gelu")
def _gelu(cfg: ModelConfig) -> nn.Module:
    if cfg.ffn.d_hidden != 4 * cfg.d_model:
        raise ValueError("ffn.type = gelu — это MLP GPT-2 с расширением ×4: d_hidden должен быть 4·d_model")
    return GPT2MLP(GPT2Config(n_embd=cfg.d_model, bias=cfg.ffn.bias))


register("ffn", "swiglu")(lambda cfg: SwiGLUFFN(cfg.d_model, cfg.ffn.d_hidden))
register("ffn", "geglu")(lambda cfg: GeGLUFFN(cfg.d_model, cfg.ffn.d_hidden))


@register("ffn", "moe")
def _moe(cfg: ModelConfig) -> nn.Module:
    m = cfg.moe
    return MoEFFN(cfg.d_model, m.d_expert, m.n_routed, m.n_shared, m.top_k, m.scoring_func, m.norm_topk_prob,
                  m.routed_scaling_factor)


# ============================================================================ блок и модель

_NORMS_BY_PLACEMENT = {
    "pre": ("input_layernorm", "post_attention_layernorm"),
    "post": ("post_attention_layernorm", "post_feedforward_layernorm"),
    "olmo": ("post_attention_layernorm", "post_feedforward_layernorm"),
    "sandwich": ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                 "post_feedforward_layernorm"),
}


class Block(nn.Module):
    """Слой: mixer + FFN с нормами по схеме C08. Имена норм — как в HF (Llama / OLMo 2 / Gemma 3).

    pre      x + f(norm(x))          GPT-2, Llama, Qwen3, DeepSeek
    post     norm(x + f(x))          оригинальный Transformer
    olmo     x + norm(f(x))          OLMo 2
    sandwich x + norm(f(norm(x)))    Gemma 2/3
    """

    def __init__(self, cfg: ModelConfig, spec: LayerSpec):
        super().__init__()
        self.placement = cfg.norm.placement
        self.spec = spec
        mixer = build("mixer", spec.mixer, cfg, spec)
        # HF Qwen3-Next: linear-слои хранят mixer в linear_attn, full — в self_attn
        self.mixer_name = "linear_attn" if spec.mixer == "gated_deltanet" else "self_attn"
        setattr(self, self.mixer_name, mixer)
        self.mlp = build("ffn", "moe" if spec.ffn == "moe" else cfg.ffn.type, cfg)
        for name in _NORMS_BY_PLACEMENT[self.placement]:
            setattr(self, name, build("norm", cfg.norm.type, cfg.d_model, cfg.norm.eps))

    @property
    def mixer(self) -> nn.Module:
        return getattr(self, self.mixer_name)

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        mix = lambda h: self.mixer(h, cache)  # noqa: E731
        if self.placement == "pre":
            x = x + mix(self.input_layernorm(x))
            return x + self.mlp(self.post_attention_layernorm(x))
        if self.placement == "post":
            x = self.post_attention_layernorm(x + mix(x))
            return self.post_feedforward_layernorm(x + self.mlp(x))
        if self.placement == "olmo":
            x = x + self.post_attention_layernorm(mix(x))
            return x + self.post_feedforward_layernorm(self.mlp(x))
        x = x + self.post_attention_layernorm(mix(self.input_layernorm(x)))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class Model(nn.Module):
    """Декодер из конфига: эмбеддинги → блоки → финальная норма → lm_head. idx [B, T] → логиты [B, T, V]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.specs = layer_specs(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.position.type == "learned":
            self.embed_positions = nn.Embedding(cfg.position.max_positions, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, spec) for spec in self.specs)
        self.norm = build("norm", cfg.norm.type, cfg.d_model, cfg.norm.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Случайная инициализация как в GPT-2 (N(0, 0.02)) — для тестов без весов. Нормы и
        параметры DeltaNet/роутера оставляем как их задали эталоны."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, name_or_cfg: str | ModelConfig) -> "Model":
        cfg = ModelConfig.named(name_or_cfg) if isinstance(name_or_cfg, str) else name_or_cfg
        return cls(cfg)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())                     # общие (tied) веса — один раз
        if exclude_embeddings:
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
            if hasattr(self, "embed_positions"):
                n -= self.embed_positions.weight.numel()
        return n

    def new_caches(self) -> list:
        """По кешу на слой: KVCache, RingKVCache (sliding), LatentCache (MLA) или состояние (DeltaNet)."""
        return [layer.mixer.new_cache() for layer in self.layers]

    def forward(self, idx: torch.Tensor, caches: list | None = None) -> torch.Tensor:
        """idx [B, T] — новые токены (после prefill — по одному) → логиты [B, T, V]."""
        T = idx.shape[1]
        x = self.embed_tokens(idx)
        if self.cfg.embed_scale:
            x = x * torch.tensor(self.cfg.d_model ** 0.5, dtype=x.dtype)
        if self.cfg.position.type == "learned":
            start = caches[0].seq_len if caches is not None else 0         # сколько токенов уже в кеше
            if start + T > self.cfg.position.max_positions:
                raise ValueError(f"Длина {start + T} больше max_positions = {self.cfg.position.max_positions}")
            x = x + self.embed_positions(torch.arange(start, start + T, device=idx.device))
        for i, layer in enumerate(self.layers):
            x = layer(x, caches[i] if caches is not None else None)
        return self.lm_head(self.norm(x))

    def router_logits(self) -> list[torch.Tensor]:
        """Логиты роутеров MoE-слоёв последнего прогона (для aux loss)."""
        return [layer.mlp.last_router_logits for layer in self.layers if isinstance(layer.mlp, MoEFFN)]


@torch.no_grad()
def generate(model: Model, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
    """Жадная генерация (как в C06): idx [B, T] → [B, T + max_new_tokens]. Сэмплирование — в C25."""
    caches = model.new_caches() if use_cache else None
    logits = model(idx, caches)
    for step in range(max_new_tokens):
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
        if step + 1 < max_new_tokens:
            logits = model(nxt, caches) if use_cache else model(idx)
    return idx
