"""SwiGLU FFN, как в Llama.

Реализуй:
1. silu(x) — активация SiLU (Swish с β = 1);
2. llama_ffn_hidden_dim(...) — правило Llama для скрытого размера FFN;
3. SwiGLUFFN.forward — FFN с гейтом.

Нельзя использовать F.silu и nn.SiLU.
"""

import torch
import torch.nn as nn


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (она же Swish с β = 1). Форма не меняется."""
    # TODO: одна строка — вход, умноженный на «насколько он открыт» (какой функцией?).
    raise NotImplementedError("silu")


def llama_ffn_hidden_dim(d_model: int, multiple_of: int = 256, ffn_dim_multiplier: float | None = None) -> int:
    """Скрытый размер FFN по правилу Llama."""
    # TODO 1: начни с 4·d_model, как в GPT-2, и возьми от него 2/3 (почему 2/3 — см. README).
    #         Приведи к int.
    # TODO 2: если задан ffn_dim_multiplier — умножь и снова приведи к int.
    # TODO 3: округли ВВЕРХ до ближайшего кратного multiple_of (целочисленной арифметикой).
    raise NotImplementedError("llama_ffn_hidden_dim")


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # Конструктор готов: имена параметров как в Llama (тесты загружают веса эталона).
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)   # ветка, через которую идёт SiLU
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)     # линейная ветка
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)   # обратно в d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] → [B, T, C]."""
        # TODO 1: две параллельные проекции входа в скрытое пространство F.
        # TODO 2: к какой из них применяется нелинейность? Как ветки объединяются?
        # TODO 3: верни результат в d_model.
        raise NotImplementedError("SwiGLUFFN.forward")
