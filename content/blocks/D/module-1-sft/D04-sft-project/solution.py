"""Проект: SFT игрушечной модели на синтетических инструкциях — что меняет SFT и что он берёт у предобучения.

Эталон. Пользователь пишет четыре функции: ``build_example``, ``check_answer``, ``chat_answer``, ``evaluate``.
Остальное (данные, «предобучение», цикл SFT, запись визуализации) — готово, запуск:

    python content/blocks/D/module-1-sft/D04-sft-project/solution.py            # ~4 минуты на CPU
    python …/solution.py --record content/viz/sft-curves.data.json              # перезаписать данные V-sft-curves
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time

import torch

from llmlab.core.hf import generate
from llmlab.core.model import Model
from llmlab.core.sft import (IGNORE, CharTokenizer, assistant_header, assistant_mask, collate, encode_chat,
                             make_instructions, masked_ce_sum, plain_text, render_chat, sft_step, sft_targets,
                             to_messages, toy_config)

# ---------------------------------------------------------------- что пишет пользователь


def build_example(example: dict, tok, loss_on: str = "response") -> tuple[list[int], torch.Tensor]:
    """Пример ``{"task", "prompt", "answer"}`` → (ids диалога «вопрос + ответ» по шаблону D01, targets D02).

    ``loss_on="response"`` — метки только у ответа ассистента; ``"all"`` — у всех токенов строки (как при
    предобучении: модель учится предсказывать и вопрос, и служебные токены).
    """
    if loss_on not in ("response", "all"):
        raise ValueError("loss_on — 'response' или 'all'")
    ids = encode_chat(render_chat(to_messages(example)), tok)
    if loss_on == "response":
        mask = assistant_mask(ids, assistant_header(tok), tok.eos_id)
    else:
        mask = torch.ones(len(ids), dtype=torch.bool)
    return ids, sft_targets(ids, mask)


def check_answer(example: dict, answer: str) -> tuple[bool, bool]:
    """(формат верен, ответ верен) для ответа модели ``answer`` (без служебных токенов).

    Формат: arith — целое число из цифр; upper — непустое слово только из заглавных букв; json — объект ровно с
    ключами name (строка) и age (целое). Ответ: формат верен и значение совпадает с эталоном (для JSON сравниваются
    объекты, а не строки: пробелы и порядок ключей не важны).
    """
    task, want = example["task"], example["answer"]
    if task == "arith":
        ok = answer.isdigit() and answer.isascii()
        return ok, ok and int(answer) == int(want)
    if task == "upper":
        ok = answer.isalpha() and answer.isupper()
        return ok, ok and answer == want
    if task == "json":
        try:
            obj = json.loads(answer)
        except ValueError:
            return False, False
        ok = (isinstance(obj, dict) and set(obj) == {"name", "age"} and isinstance(obj["name"], str)
              and isinstance(obj["age"], int) and not isinstance(obj["age"], bool))
        return ok, ok and obj == json.loads(want)
    raise ValueError(f"неизвестная задача {task!r}")


@torch.no_grad()
def chat_answer(model, tok, prompt: str, max_new_tokens: int = 40) -> str:
    """Ответ модели на вопрос в chat-формате: generation prompt без рассуждения, жадно до ``<|im_end|>``."""
    text = render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
    ids = encode_chat(text, tok)
    out = generate(model, torch.tensor([ids]), max_new_tokens, eos_token_id=tok.eos_id)[0, len(ids):].tolist()
    if tok.eos_id in out:
        out = out[:out.index(tok.eos_id)]
    return tok.decode(out)


def evaluate(examples: list[dict], answer_fn) -> dict:
    """Доли верного формата и верного ответа: в целом (``format``, ``exact``) и по задачам (``by_task``)."""
    by_task: dict[str, list[tuple[bool, bool]]] = {}
    for ex in examples:
        by_task.setdefault(ex["task"], []).append(check_answer(ex, answer_fn(ex["prompt"])))
    flat = [r for rs in by_task.values() for r in rs]
    share = lambda rs, k: sum(r[k] for r in rs) / len(rs) if rs else 0.0
    return {"format": share(flat, 0), "exact": share(flat, 1),
            "by_task": {t: {"format": share(rs, 0), "exact": share(rs, 1)} for t, rs in by_task.items()}}


# ---------------------------------------------------------------- готовое: эксперимент


def pretrain(model, tok, steps: int, seed: int = 0, batch: int = 16, docs_per_row: int = 5, lr: float = 3e-3) -> list:
    """«Предобучение» на сплошном тексте: строки из нескольких фактов подряд, лосс по всем токенам."""
    rng = random.Random(seed)
    facts = [plain_text(e) for e in make_instructions(60_000, seed=seed + 2, split="all")]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps):
        rows = []
        for _ in range(batch):
            x = tok.encode("".join(rng.choice(facts) for _ in range(docs_per_row)))[:256]
            rows.append((x, x[1:] + [IGNORE]))
        st = sft_step(model, opt, [collate(rows, tok.pad_id)], step, max_lr=lr, warmup_steps=50, total_steps=steps,
                      min_lr=lr / 10)
        history.append(st["loss"])
    return history


@torch.no_grad()
def response_loss(model, data: list[tuple], pad_id: int) -> float:
    """Средний лосс на токен ответа (teacher forcing) — одна шкала для всех режимов обучения."""
    model.eval()
    s, n = 0.0, 0
    for i in range(0, len(data), 64):
        ids, tgt = collate(data[i:i + 64], pad_id)
        ls, k = masked_ce_sum(model(ids), tgt)
        s, n = s + float(ls), n + k
    model.train()
    return s / n


def run_sft(model, tok, train: list[dict], val: list[dict], test: list[dict], loss_on: str, steps: int,
            lr: float, eval_every: int, seed: int = 1, batch: int = 16) -> dict:
    data = [build_example(e, tok, loss_on) for e in train]
    val_data = [build_example(e, tok, "response") for e in val]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    log = {"steps": [], "val_loss": [], "format": [], "exact": [], "by_task": []}

    def record(step):
        model.eval()
        res = evaluate(test, lambda p: chat_answer(model, tok, p))
        model.train()
        log["steps"].append(step)
        log["val_loss"].append(round(response_loss(model, val_data, tok.pad_id), 4))
        log["format"].append(round(res["format"], 4))
        log["exact"].append(round(res["exact"], 4))
        log["by_task"].append(res["by_task"])

    record(0)
    for step in range(steps):
        idx = torch.randint(len(data), (batch,), generator=g).tolist()
        sft_step(model, opt, [collate([data[i] for i in idx], tok.pad_id)], step, max_lr=lr, warmup_steps=10,
                 total_steps=steps, min_lr=lr / 10)
        if (step + 1) % eval_every == 0:
            record(step + 1)
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT игрушечной модели: база против «с нуля», лосс по ответу против всех токенов")
    ap.add_argument("--pretrain-steps", type=int, default=3000)
    ap.add_argument("--sft-steps", type=int, default=300)
    ap.add_argument("--sft-examples", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--record", type=str, default=None, help="записать JSON для визуализации sft-curves")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = CharTokenizer()
    cfg = toy_config(tok.vocab_size, args.d_model, args.layers)
    base = Model(cfg)
    scratch = copy.deepcopy(base)                    # та же инициализация, но без предобучения
    train = make_instructions(args.sft_examples, seed=0)
    val = make_instructions(150, seed=3, split="test")
    test = make_instructions(150, seed=1, split="test")

    t0 = time.time()
    pre_hist = pretrain(base, tok, args.pretrain_steps)
    print(f"предобучение: {args.pretrain_steps} шагов, лосс {pre_hist[0]:.2f} → {pre_hist[-1]:.3f}, "
          f"{time.time() - t0:.0f} с, {base.num_params():,} параметров".replace(",", " "))
    base.eval()
    plain = evaluate(test, lambda p: _plain_answer(base, tok, p))
    print(f"база, сплошной текст «… Ответ: …»: формат {plain['format']:.2f}, ответ {plain['exact']:.2f}")
    sample = test[0]
    print(f"база в chat-формате на «{sample['prompt']}»: {chat_answer(base, tok, sample['prompt'])!r}")
    base.train()

    runs = {}
    for name, init, loss_on in [("base_response", base, "response"), ("base_all", base, "all"),
                                ("scratch_response", scratch, "response")]:
        t1 = time.time()
        model = copy.deepcopy(init)
        runs[name] = run_sft(model, tok, train, val, test, loss_on, args.sft_steps, args.lr, args.eval_every)
        r = runs[name]
        print(f"{name:17s} {time.time() - t1:4.0f} с: формат {r['format'][0]:.2f} → {r['format'][-1]:.2f}, "
              f"ответ {r['exact'][0]:.2f} → {r['exact'][-1]:.2f}, лосс ответа {r['val_loss'][-1]:.3f}, "
              f"по задачам {json.dumps({t: v['exact'] for t, v in r['by_task'][-1].items()}, ensure_ascii=False)}")
        if name == "base_response":
            for ex in test[:6]:
                print(f"   {ex['prompt']!r} → {chat_answer(model, tok, ex['prompt'])!r} (ждём {ex['answer']!r})")

    if args.record:
        out = {"config": {"d_model": args.d_model, "layers": args.layers, "params": base.num_params(),
                          "pretrain_steps": args.pretrain_steps, "sft_steps": args.sft_steps,
                          "sft_examples": args.sft_examples, "lr": args.lr, "test_examples": len(test)},
               "plain": {"format": plain["format"], "exact": plain["exact"], "by_task": plain["by_task"]},
               "runs": runs}
        with open(args.record, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"записано: {args.record}")


@torch.no_grad()
def _plain_answer(model, tok, prompt: str) -> str:
    """Ответ базы в её «родном» формате: факт сплошным текстом «… Ответ: X.» после другого факта."""
    ids = tok.encode(prompt + " Ответ: ")
    out = tok.decode(generate(model, torch.tensor([ids]), 40)[0, len(ids):].tolist())
    return out.split(".\n")[0] if ".\n" in out else out.split("\n")[0].rstrip(".")


if __name__ == "__main__":
    main()
