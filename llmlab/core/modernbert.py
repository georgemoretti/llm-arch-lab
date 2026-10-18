"""ModernBERT и эмбеддинги текста — эталоны задач B16–B19 (блок B, модуль 4).

Задачи B17–B20 импортируют готовые части отсюда, а не из кода пользователя. Совпадение функций и классов с
``solution.py`` задач проверяет ``tests/test_core_modernbert.py``.

Имена параметров — как в Hugging Face ``ModernBertModel``: state_dict из ``transformers`` загружается без
переименований. RoPE — из C03 (``core/llama.py``, раскладка «половинами», как в HF). Маски — bool, True = «можно
смотреть». Обозначения: B — batch, T — длина, C — hidden_size, H — головы, D — размер головы, N — число настоящих
токенов в упакованном батче.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, rope_cos_sin

# ---------------------------------------------------------------- B16: слой ModernBERT


@dataclass
class ModernBertConfig:
    """Гиперпараметры ModernBERT. По умолчанию — игрушечные размеры; окно, шаг и θ — как у ModernBERT-base."""

    vocab_size: int
    hidden_size: int = 32
    intermediate_size: int = 48
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    global_attn_every_n_layers: int = 3
    local_attention: int = 128                 # ширина окна: ±64 токена
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    pad_token_id: int = 0


def layer_types(num_layers: int, global_every: int) -> list[str]:
    """Тип attention каждого слоя: слой 0 и каждый global_every-й — глобальный, остальные — локальные."""
    return ["full_attention" if i % global_every == 0 else "sliding_attention" for i in range(num_layers)]


def local_window_mask(T: int, window: int, device=None) -> torch.Tensor:
    """[T, T] bool: запрос i видит ключ j, если |i − j| ≤ window // 2 (окно window = 128 → по 64 в каждую сторону)."""
    pos = torch.arange(T, device=device)
    return (pos[:, None] - pos[None, :]).abs() <= window // 2


class ModernBertEmbeddings(nn.Module):
    """Только эмбеддинги токенов и LayerNorm без bias: позиции добавит RoPE в каждом слое."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU: Wi даёт сразу две половины [вход | gate], выход Wo(GELU(вход) · gate). Без bias."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(inp) * gate)


class ModernBertAttention(nn.Module):
    """Двунаправленное attention с RoPE: одна матрица Wqkv, без bias; θ RoPE зависит от типа слоя."""

    def __init__(self, cfg: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention_type = layer_types(cfg.num_hidden_layers, cfg.global_attn_every_n_layers)[layer_idx]
        self.rope_theta = cfg.global_rope_theta if self.attention_type == "full_attention" else cfg.local_rope_theta
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, C], mask — bool, транслируется к [B, H, T, T] (True = можно смотреть) → [B, T, C]."""
        B, T, C = x.shape
        qkv = self.Wqkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                                 # три тензора [B, H, T, D]
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), self.head_dim, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        if mask is not None:
            # finfo.min, а не −inf: у PAD-запроса локальное окно может целиком состоять из PAD → softmax(−inf…) = nan
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        out = scores.softmax(-1) @ v
        return self.Wo(out.transpose(1, 2).reshape(B, T, C))


class ModernBertEncoderLayer(nn.Module):
    """Pre-LN: x + attn(attn_norm(x)) → x + mlp(mlp_norm(x)). У слоя 0 attn_norm = Identity: вход уже нормирован."""

    def __init__(self, cfg: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = (nn.Identity() if layer_idx == 0
                          else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False))
        self.attn = ModernBertAttention(cfg, layer_idx)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.mlp = ModernBertMLP(cfg)
        self.attention_type = self.attn.attention_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertModel(nn.Module):
    """Эмбеддинги → слои (глобальные и локальные вперемежку) → final_norm: [B, T] → [B, T, C]."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = ModernBertEmbeddings(cfg)
        self.layers = nn.ModuleList(ModernBertEncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pad = attention_mask.bool()[:, None, None, :]                       # [B, 1, 1, T]: PAD-ключи закрыты
        masks = {"full_attention": pad,
                 "sliding_attention": pad & local_window_mask(T, self.cfg.local_attention, input_ids.device)}
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, masks[layer.attention_type])
        return self.final_norm(x)


class ModernBertForMaskedLM(nn.Module):
    """ModernBERT + MLM-голова: dense → GELU → LayerNorm → decoder (матрица = tok_embeddings, свой bias)."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.model = ModernBertModel(cfg)
        self.head = nn.Module()
        self.head.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.head.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.model(input_ids, attention_mask)
        return self.decoder(self.head.norm(F.gelu(self.head.dense(h))))


# ---------------------------------------------------------------- B17: unpadding


def unpad_input(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """attention_mask [B, T] (1 / 0) → (indices [N] — номера настоящих токенов в x.view(B·T, …),
    cu_seqlens [B + 1] int32 — границы текстов в упаковке, max_seqlen)."""
    seqlens = attention_mask.sum(-1, dtype=torch.int32)
    indices = attention_mask.flatten().nonzero().flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, int(seqlens.max())


def pack(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """[B, T, …] → [N, …]: только настоящие токены, текст за текстом."""
    return x.flatten(0, 1)[indices]


def unpack(packed: torch.Tensor, indices: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """[N, …] → [B, T, …]; на месте PAD — нули."""
    out = packed.new_zeros(B * T, *packed.shape[1:])
    out[indices] = packed
    return out.view(B, T, *packed.shape[1:])


def packed_position_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Позиции в упаковке [N]: у каждого текста свои 0, 1, 2, … (для RoPE)."""
    N = int(cu_seqlens[-1])
    seq_id = torch.searchsorted(cu_seqlens[1:], torch.arange(N, device=cu_seqlens.device), right=True)
    return torch.arange(N, device=cu_seqlens.device) - cu_seqlens[seq_id].long()


def varlen_mask(cu_seqlens: torch.Tensor, window: int | None = None) -> torch.Tensor:
    """[N, N] bool, блочно-диагональная: токен видит только токены своего текста (и, если задано, в окне ±window // 2)."""
    N = int(cu_seqlens[-1])
    pos = torch.arange(N, device=cu_seqlens.device)
    seq_id = torch.searchsorted(cu_seqlens[1:], pos, right=True)
    mask = seq_id[:, None] == seq_id[None, :]
    if window is not None:
        mask = mask & ((pos[:, None] - pos[None, :]).abs() <= window // 2)
    return mask


def varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                     window: int | None = None) -> torch.Tensor:
    """q, k, v [N, H, D] упакованного батча → [N, H, D]; тексты не видят друг друга."""
    D = q.shape[-1]
    scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1) / D ** 0.5          # [H, N, N]
    scores = scores.masked_fill(~varlen_mask(cu_seqlens, window), torch.finfo(scores.dtype).min)
    return (scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1)


def padding_savings(attention_mask: torch.Tensor) -> dict[str, float]:
    """Какую долю работы на паддинг экономит упаковка: линейные слои ∝ числу токенов, attention ∝ Σ L²."""
    lengths = attention_mask.sum(-1).double()
    B, T = attention_mask.shape
    return {"linear": 1 - float(lengths.sum()) / (B * T),
            "attention": 1 - float((lengths ** 2).sum()) / (B * T * T)}


# ---------------------------------------------------------------- B18: sentence-эмбеддинги


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Среднее по настоящим токенам: [B, T, C], [B, T] → [B, C]. PAD в среднее не входит."""
    m = attention_mask.to(hidden.dtype).unsqueeze(-1)                   # [B, T, 1]
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Вектор первого токена ([CLS]): [B, T, C] → [B, C]."""
    return hidden[:, 0]


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """x / ‖x‖ по последней оси; нулевой вектор остаётся нулевым."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Косинусы всех пар: [N, C], [M, C] → [N, M]."""
    return l2_normalize(a) @ l2_normalize(b).T


def _hidden(encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Скрытые состояния [B, T, C] любого энкодера курса: ModernBertModel возвращает тензор, BertModel — (h, pooled)."""
    kw = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
    out = encoder(input_ids, attention_mask=attention_mask, **kw)
    return out[0] if isinstance(out, tuple) else out


class BiEncoder(nn.Module):
    """Bi-encoder: каждый текст кодируется отдельно в один вектор; близость — косинус (скалярное после L2)."""

    def __init__(self, encoder: nn.Module, pooling: str = "mean", normalize: bool = True):
        super().__init__()
        assert pooling in ("mean", "cls"), pooling
        self.encoder, self.pooling, self.normalize = encoder, pooling, normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = _hidden(self.encoder, input_ids, attention_mask)
        e = mean_pool(h, attention_mask) if self.pooling == "mean" else cls_pool(h)
        return l2_normalize(e) if self.normalize else e


class CrossEncoder(nn.Module):
    """Cross-encoder: запрос и документ — один вход [CLS] q [SEP] d [SEP]; оценка — Linear по вектору [CLS]."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = _hidden(self.encoder, input_ids, attention_mask, token_type_ids)
        return self.classifier(cls_pool(h)).squeeze(-1)                 # [B]
