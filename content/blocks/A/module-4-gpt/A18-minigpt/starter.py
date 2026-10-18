"""Мини-GPT на символах корпуса курса: модель, батчи, обучение, генерация.

Реализуй MiniGPT.forward, MiniGPT.generate, get_batch и train. Конструктор и всё, что ниже train, готово.
Детали из прошлых задач (эмбеддинги, блок) импортированы из llmlab.core.minigpt.
Обучение целиком (~30 с на CPU):  python workspace/A18-minigpt/solution.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.minigpt import Block, TokenPositionEmbedding, tie_weights   # эталоны A15–A17

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"


@dataclass
class MiniGPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_embd: int = 64
    pre_norm: bool = True


class MiniGPT(nn.Module):
    """Эмбеддинги → n_layer блоков → финальный LayerNorm → выходной слой, связанный с эмбеддингами."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        # Готово, не меняй: имена слоёв проверяют тесты (они загружают веса эталона).
        self.config = config
        self.emb = TokenPositionEmbedding(config.vocab_size, config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config.n_embd, config.block_size, config.pre_norm)
                                    for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        tie_weights(self.lm_head, self.emb.wte)
        for name, p in self.named_parameters():          # как в GPT-2: N(0, 0.02²), выходные проекции меньше
            if name.endswith("weight") and p.dim() == 2 and "emb" not in name:
                std = 0.02 / math.sqrt(2 * config.n_layer) if name.endswith("c_proj.weight") else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx [B, T] → (logits [B, T, V], loss или None)."""
        # TODO 1: эмбеддинги (self.emb) → все блоки по очереди → финальный LayerNorm → lm_head.
        # TODO 2: loss — как в A15: cross_entropy по всем B·T позициям; без targets — None.
        raise NotImplementedError("MiniGPT.forward")

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Дописать max_new_tokens токенов: [B, T] → [B, T + max_new_tokens]. temperature = 0 → жадный выбор."""
        # TODO 1: режим eval на время генерации (потом вернуть прежний: self.training).
        # TODO 2: max_new_tokens раз: контекст — не больше последних block_size токенов (почему?),
        #         логиты последней позиции [B, V], следующий токен:
        #         temperature == 0 → argmax; иначе torch.multinomial(softmax(логиты / temperature), 1,
        #         generator=generator). Допиши его к idx (torch.cat по оси времени).
        raise NotImplementedError("MiniGPT.generate")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator | None = None):
    """Случайные куски текста: x [B, T] и y [B, T], где y — тот же кусок, сдвинутый на один токен вперёд."""
    # TODO 1: batch_size случайных начал s: torch.randint(0, ?, (batch_size,), generator=generator).
    #         Какое самое правое начало ещё даёт полный x и полный y (y заканчивается на токен s + block_size)?
    # TODO 2: x[i] = data[s : s + block_size], y[i] — тот же кусок, сдвинутый на 1. torch.stack по батчу.
    raise NotImplementedError("get_batch")


def train(model: MiniGPT, data: torch.Tensor, steps: int, batch_size: int = 32, lr: float = 1e-3,
          generator: torch.Generator | None = None) -> list[float]:
    """AdamW, lr постоянный: steps шагов на случайных батчах; возвращает лосс каждого шага."""
    # TODO: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01), модель в режиме train.
    #       Каждый шаг: get_batch(data, model.config.block_size, batch_size, generator) → loss → zero_grad
    #       → backward → clip_grad_norm_(…, 1.0) → step. Верни список loss.item() всех шагов.
    #       Порядок и параметры важны: тест сверяет историю лоссов с эталоном.
    raise NotImplementedError("train")


# ---------------------------------------------------------------------------- запуск целиком (не тесты)


@torch.no_grad()
def estimate_loss(model: MiniGPT, data: torch.Tensor, batches: int = 4) -> float:
    """Средний лосс на нескольких фиксированных батчах (режим eval, затем прежний режим)."""
    was_training = model.training
    model.eval()
    losses = [model(*get_batch(data, model.config.block_size, 64, torch.Generator().manual_seed(i)))[1].item()
              for i in range(batches)]
    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser(description="Обучить мини-GPT на корпусе курса и показать генерацию.")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--every", type=int, default=100, help="как часто считать val-лосс и печатать образец текста")
    p.add_argument("--patience", type=int, default=3, help="остановиться, если val не улучшался столько проверок подряд")
    p.add_argument("--record", type=Path, default=None, help="сохранить кривую и образцы в JSON (для визуализации)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    text = CORPUS.read_text(encoding="utf-8")
    itos = sorted(set(text))
    stoi = {c: i for i, c in enumerate(itos)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]          # последние 10% текста модель не видит

    model = MiniGPT(MiniGPTConfig(vocab_size=len(itos)))
    print(f"словарь {len(itos)} символов, текст {len(text)} символов, параметров {sum(q.numel() for q in model.parameters()):,}")
    print(f"стартовый лосс {estimate_loss(model, val_data):.3f} (ln V = {math.log(len(itos)):.3f})")
    g = torch.Generator().manual_seed(args.seed)
    prompt = torch.tensor([[stoi[c] for c in "Модель "]])
    record = {"steps": [], "train": [], "val": [], "samples": []}
    best_val, best_state, best_step, bad = float("inf"), None, 0, 0
    t0 = time.time()
    done = 0
    while done < args.steps:
        losses = train(model, train_data, min(args.every, args.steps - done), generator=g)
        done += len(losses)
        val = estimate_loss(model, val_data)
        sample = "".join(itos[i] for i in model.generate(prompt, 120, 0.8, torch.Generator().manual_seed(2))[0])
        tr = sum(losses[-50:]) / len(losses[-50:])
        print(f"\nшаг {done:5d}  train {tr:.3f}  val {val:.3f}  ({time.time() - t0:.0f} с)\n{sample}")
        record["steps"].append(done)
        record["train"].append(round(tr, 4))
        record["val"].append(round(val, 4))
        record["samples"].append(sample)
        if val < best_val:
            best_val, best_step, bad = val, done, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience and not args.record:   # для записи кривой учим до конца: видно переобучение
                print(f"\nval не улучшался {bad} проверки подряд — модель начала заучивать текст, останавливаемся")
                break
    model.load_state_dict(best_state)
    print(f"\nлучший val {best_val:.3f} на шаге {best_step}. Генерация лучшей модели (temperature 0.8):\n")
    print("".join(itos[i] for i in model.generate(prompt, 400, 0.8, torch.Generator().manual_seed(3))[0]))
    if args.record:
        record["best_step"] = best_step
        record["init_loss"] = round(math.log(len(itos)), 4)
        record["config"] = {"vocab_size": len(itos), **{k: v for k, v in vars(model.config).items() if k != "vocab_size"},
                            "params": sum(q.numel() for q in model.parameters()), "lr": 1e-3, "batch_size": 32}
        args.record.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"записано в {args.record}")


if __name__ == "__main__":
    main()
