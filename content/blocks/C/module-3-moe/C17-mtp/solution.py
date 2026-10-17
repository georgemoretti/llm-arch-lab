"""Эталон: Multi-Token Prediction (DeepSeek-V3) — модули предсказания t+2, …, t+D+1 и их loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.gemma_olmo import PreNormBlock   # блок трансформера (Pre-Norm, как Llama), эталон «Место нормализации»
from llmlab.core.llama import RMSNorm             # эталон «RMSNorm»


class MTPModule(nn.Module):
    """Один модуль Multi-Token Prediction DeepSeek-V3: смешать h предыдущей глубины и эмбеддинг
    следующего токена, прогнать через блок трансформера."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_hidden: int):
        super().__init__()
        self.enorm = RMSNorm(d_model)
        self.hnorm = RMSNorm(d_model)
        self.eh_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = PreNormBlock(d_model, n_heads, n_kv_heads, d_hidden)
        self.norm = RMSNorm(d_model)                     # норма перед общей выходной головой

    def forward(self, h_prev: torch.Tensor, emb_next: torch.Tensor) -> torch.Tensor:
        """h_prev, emb_next [B, T', C] → h [B, T', C]. Порядок конкатенации — как в vLLM: [emb; h]."""
        x = self.eh_proj(torch.cat((self.enorm(emb_next), self.hnorm(h_prev)), dim=-1))
        return self.block(x)


def mtp_logits(h_main: torch.Tensor, tokens: torch.Tensor, embed: nn.Embedding, lm_head: nn.Module,
               modules: nn.ModuleList) -> list[torch.Tensor]:
    """h_main [B, T, C] (выход основной модели), tokens [B, T] → логиты глубин k = 1..D, [B, T−k, V].

    Позиция i глубины k видит токены ≤ i + k и предсказывает токен i + k + 1.
    """
    T = tokens.shape[1]
    h, out = h_main, []
    for k, module in enumerate(modules, start=1):
        h = module(h[:, : T - k], embed(tokens[:, k:]))                  # [B, T−k, C]
        out.append(lm_head(module.norm(h)))
    return out


def mtp_loss(main_logits: torch.Tensor, mtp_logits_list: list[torch.Tensor], tokens: torch.Tensor, lam: float):
    """→ (L_main + λ/D · Σ L_k, L_main, [L_k]). Основная голова: i → i+1; глубина k: i → i+k+1."""
    V = main_logits.shape[-1]
    l_main = F.cross_entropy(main_logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
    l_mtp = [F.cross_entropy(lg[:, :-1].reshape(-1, V), tokens[:, k + 1:].reshape(-1))
             for k, lg in enumerate(mtp_logits_list, start=1)]
    D = len(l_mtp)
    total = l_main + (lam / D) * sum(l_mtp) if D else l_main
    return total, l_main, l_mtp
