"""Проект: мини-поиск по корпусу курса — bi-encoder на ModernBERT против мешка слов (TF-IDF).

Реализуй пять функций из раздела «твоя часть»: ict_pair, idf_weights, tfidf_vectors, train_step, hit_at_k.
Остальное готово. Корпус — абзацы README задач курса, запросы для оценки — вопросы курса (снимок в
content/data/course-search.json). Обучение без разметки: пары «фрагмент абзаца ↔ остальной абзац» (inverse cloze task).

    python solution.py                      # обучить (~1.5 мин на CPU) и сравнить с TF-IDF
    python solution.py --record PATH        # ещё записать данные для визуализации embedding-space
    python solution.py --pretrained         # ещё сравнить с BAAI/bge-m3 (~2.3 ГБ, нужен transformers)
    python solution.py --rebuild-corpus     # пересобрать снимок корпуса из текущих README (числа в README задачи изменятся)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
from pathlib import Path

import torch

from llmlab.core.modernbert import BiEncoder, ModernBertConfig, ModernBertModel, info_nce
from llmlab.paths import project_root

# корень курса: вверх от файла (и из workspace/), иначе — LLMLAB_ROOT / текущая папка (понятная ошибка, если нет)
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "content" / "data").is_dir()), None) or project_root()
CORPUS = ROOT / "content" / "data" / "course-search.json"
PAD, UNK, CLS = 0, 1, 2

# ---------------------------------------------------------------- корпус и токенизатор (готово)


def split_paragraphs(markdown: str, min_words: int = 15) -> list[str]:
    """Абзацы README без кода, формул, таблиц, заголовков и директив; разметка Markdown снята."""
    md = re.sub(r"```.*?```", "\n\n", markdown, flags=re.S)
    md = re.sub(r"\$\$.*?\$\$", " ", md, flags=re.S)
    out = []
    for block in re.split(r"\n\s*\n", md):
        lines = [ln for ln in block.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "|", ":::"))]
        text = " ".join(ln.strip().lstrip("-*0123456789. ").strip() for ln in lines)
        text = re.sub(r"\$[^$]*\$", " ", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\s+", " ", re.sub(r"[`*_]", "", text)).strip()
        if len(text.split()) >= min_words:
            out.append(text)
    return out


def build_corpus(root: Path = ROOT) -> dict:
    """Снимок: абзацы README всех задач и вопросы курса (source: course), у которых есть задача с абзацами."""
    import yaml

    paragraphs = [{"task": p.parent.name, "text": t}
                  for p in sorted(root.glob("content/blocks/*/*/*/README.md")) for t in split_paragraphs(p.read_text())]
    tasks = {p["task"] for p in paragraphs}
    questions = []
    for f in sorted(root.glob("content/questions/*.yaml")):
        for q in yaml.safe_load(f.read_text(encoding="utf-8")) or []:
            if q.get("source") == "course" and set(q.get("tasks") or []) & tasks:
                questions.append({"id": q["id"], "tasks": [t for t in q["tasks"] if t in tasks], "text": q["question"]})
    return {"paragraphs": paragraphs, "questions": questions}


def load_corpus(path: Path = CORPUS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


WORD = re.compile(r"[a-zа-я0-9]+")


def words(text: str) -> list[str]:
    """Слова в нижнем регистре (ё → е), без пунктуации."""
    return WORD.findall(text.lower().replace("ё", "е"))


def pieces(text: str, stem: int = 5) -> list[str]:
    """Грубая морфология: слово → «основа» из первых 5 букв и «##окончание». «маскирование» → маски ##рование."""
    out = []
    for w in words(text):
        out.append(w[:stem])
        if len(w) > stem:
            out.append("##" + w[stem:])
    return out


def build_vocab(texts: list[str], min_count: int = 2) -> dict[str, int]:
    """Словарь кусков: [PAD] = 0, [UNK] = 1, [CLS] = 2, дальше куски, встретившиеся ≥ min_count раз."""
    counts = collections.Counter(p for t in texts for p in pieces(t))
    itos = ["[PAD]", "[UNK]", "[CLS]"] + sorted(p for p, c in counts.items() if c >= min_count)
    return {p: i for i, p in enumerate(itos)}


def encode(text: str, stoi: dict[str, int], max_len: int) -> list[int]:
    return ([CLS] + [stoi.get(p, UNK) for p in pieces(text)])[:max_len]


def collate(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Список id → input_ids [B, T] (PAD справа) и attention_mask [B, T]."""
    ids = torch.full((len(seqs), max(map(len, seqs))), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    return ids, (ids != PAD).long()


# ---------------------------------------------------------------- твоя часть


def ict_pair(text_words: list[str], rng: random.Random, min_len: int = 6, max_len: int = 14,
             remove_prob: float = 0.9) -> tuple[list[str], list[str]]:
    """Inverse cloze task: запрос — случайный кусок абзаца из min_len … max_len слов подряд; контекст — абзац,
    из которого кусок с вероятностью remove_prob вырезан (иначе — абзац целиком)."""
    # TODO 1: длина куска n = rng.randint(min_len, max_len); начало start = rng.randrange(...) так, чтобы кусок
    #         целиком лежал в абзаце (если абзац короче n — start = 0).
    # TODO 2: query = n слов подряд с позиции start.
    # TODO 3: с вероятностью remove_prob (rng.random() < remove_prob) контекст — абзац БЕЗ этого куска, иначе — весь
    #         абзац (копия списка). Вызовы rng — ровно в этом порядке: randint, randrange, random.
    raise NotImplementedError("ict_pair")


def idf_weights(docs: list[list[str]], vocab: dict[str, int]) -> torch.Tensor:
    """Сглаженный idf [V]: log((1 + N) / (1 + df)) + 1, df — число документов, где термин встретился."""
    # TODO: df[v] — в скольких документах термин v встретился хотя бы раз (set по документу; термины вне словаря
    #       пропусти); idf = log((1 + N) / (1 + df)) + 1 — тензор [V].
    raise NotImplementedError("idf_weights")


def tfidf_vectors(texts: list[list[str]], vocab: dict[str, int], idf: torch.Tensor) -> torch.Tensor:
    """Мешок слов: счётчик термина × idf, L2-нормировка строки → [N, V]; термины вне словаря пропускаются."""
    # TODO 1: матрица [N, V] из счётчиков: сколько раз термин встретился в тексте i (вне словаря — пропустить).
    # TODO 2: умножить на idf (по столбцам).
    # TODO 3: L2-нормировать строки (пустой текст — нулевая строка, без nan: clamp(min=1e-12)).
    raise NotImplementedError("tfidf_vectors")


def train_step(model: BiEncoder, opt: torch.optim.Optimizer, q_ids: torch.Tensor, q_mask: torch.Tensor,
               d_ids: torch.Tensor, d_mask: torch.Tensor, temperature: float = 0.05) -> float:
    """Один шаг: эмбеддинги запросов и документов батча → InfoNCE с in-batch негативами → шаг оптимизатора."""
    # TODO: эмбеддинги запросов и документов (model(ids, mask), уже L2-нормированы); info_nce из B19 с temperature;
    #       zero_grad → backward → step. Вернуть лосс как float.
    raise NotImplementedError("train_step")


def hit_at_k(scores: torch.Tensor, gold: list[set[int]], k: int) -> float:
    """Доля запросов, у которых среди k лучших документов есть хотя бы один правильный (gold[i] — их номера)."""
    # TODO: k лучших документов каждого запроса (topk; k не больше числа документов); запрос «попал», если среди
    #       них есть хоть один из gold[i]. Вернуть долю попаданий.
    raise NotImplementedError("hit_at_k")


# ---------------------------------------------------------------- модель, оценка, обучение (готово)


def make_model(vocab_size: int, seed: int = 0) -> BiEncoder:
    torch.manual_seed(seed)
    cfg = ModernBertConfig(vocab_size=vocab_size, hidden_size=96, intermediate_size=192, num_hidden_layers=2,
                           num_attention_heads=4, local_attention=32)
    return BiEncoder(ModernBertModel(cfg), pooling="mean")


@torch.no_grad()
def embed(model: BiEncoder, texts: list[str], stoi: dict[str, int], max_len: int, batch: int = 128) -> torch.Tensor:
    was = model.training
    model.eval()
    out = [model(*collate([encode(t, stoi, max_len) for t in texts[i:i + batch]])) for i in range(0, len(texts), batch)]
    model.train(was)
    return torch.cat(out)


def eval_sets(corpus: dict, n_fragments: int = 300, seed: int = 0):
    """Два набора запросов: отложенные фрагменты из 8 слов (лексический поиск) и вопросы курса (смысловой)."""
    rng = random.Random(seed)
    paras = corpus["paragraphs"]
    frags = []
    for i in rng.sample(range(len(paras)), n_fragments):
        w = paras[i]["text"].split()
        s = rng.randrange(max(1, len(w) - 8))
        frags.append((" ".join(w[s:s + 8]), {i}))
    by_task = collections.defaultdict(set)
    for i, p in enumerate(paras):
        by_task[p["task"]].add(i)
    questions = [(q["text"], set().union(*(by_task[t] for t in q["tasks"]))) for q in corpus["questions"]]
    return {"fragments": frags, "questions": questions}


def report(name: str, scores: dict[str, torch.Tensor], sets: dict, ks=(1, 5, 10)) -> dict:
    row = {s: {k: hit_at_k(scores[s], [g for _, g in sets[s]], k) for k in ks} for s in sets}
    print(f"{name:<28} фрагменты r@1 {row['fragments'][1]:.2f}   вопросы r@1 / r@5 / r@10 "
          + " / ".join(f"{row['questions'][k]:.2f}" for k in ks))
    return row


def anisotropy(emb: torch.Tensor, seed: int = 0) -> float:
    """Средний косинус случайных пар разных текстов."""
    g = torch.Generator().manual_seed(seed)
    i, j = torch.randint(len(emb), (2, 2000), generator=g)
    keep = i != j
    return float((emb[i[keep]] * emb[j[keep]]).sum(-1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--rebuild-corpus", action="store_true")
    args = ap.parse_args()
    if args.rebuild_corpus:
        CORPUS.write_text(json.dumps(build_corpus(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    corpus = load_corpus()
    paras = [p["text"] for p in corpus["paragraphs"]]
    sets = eval_sets(corpus)
    print(f"корпус: {len(paras)} абзацев, {len(corpus['questions'])} вопросов, {sum(map(len, paras))} символов")

    results = {}
    for name, tok in (("TF-IDF по словам", words), ("TF-IDF по основам", pieces)):
        docs = [tok(t) for t in paras]
        vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        idf = idf_weights(docs, vocab)
        D = tfidf_vectors(docs, vocab, idf)
        sc = {s: tfidf_vectors([tok(q) for q, _ in sets[s]], vocab, idf) @ D.T for s in sets}
        results[name] = report(name, sc, sets)
    tfidf_scores = sc                                                   # по основам — для гибрида

    stoi = build_vocab(paras)
    model = make_model(len(stoi))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = random.Random(0)
    para_words = [t.split() for t in paras]
    probe = random.Random(1).sample(range(len(paras)), 200)            # абзацы для визуализации

    def snapshot(step: int) -> dict:
        D = embed(model, paras, stoi, 128)
        sc = {s: embed(model, [q for q, _ in sets[s]], stoi, 32) @ D.T for s in sets}
        pos = [float(sc["fragments"][i, next(iter(g))]) for i, (_, g) in enumerate(sets["fragments"])]
        return {"step": step, "D": D, "scores": sc, "pos": pos, "aniso": anisotropy(D),
                "r_frag": hit_at_k(sc["fragments"], [g for _, g in sets["fragments"]], 1),
                "r_q": hit_at_k(sc["questions"], [g for _, g in sets["questions"]], 5)}

    snaps = [snapshot(0)]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pairs = [ict_pair(para_words[i], rng) for i in rng.sample(range(len(paras)), args.batch)]
        q = collate([encode(" ".join(a), stoi, 32) for a, _ in pairs])
        d = collate([encode(" ".join(b), stoi, 128) for _, b in pairs])
        loss = train_step(model, opt, *q, *d)
        if step % args.every == 0 or step == args.steps:
            snaps.append(snapshot(step))
            snaps[-1]["loss"] = loss
            print(f"шаг {step:5d}  лосс {loss:.3f}  фрагменты r@1 {snaps[-1]['r_frag']:.2f}  вопросы r@5 "
                  f"{snaps[-1]['r_q']:.2f}  средний cos {snaps[-1]['aniso']:.2f}  {time.time() - t0:.0f} с")
    final = snaps[-1]
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} М параметров, словарь {len(stoi)}")
    results["bi-encoder (ModernBERT, 1 мин)"] = report("bi-encoder (наш)", final["scores"], sets)
    hybrid = {s: 0.2 * final["scores"][s] + 0.8 * tfidf_scores[s] for s in sets}
    results["гибрид 0.2 · наш + 0.8 · TF-IDF"] = report("гибрид 0.2 · наш + 0.8 · TF-IDF", hybrid, sets)

    if args.pretrained:
        import transformers as tr
        m = tr.AutoModel.from_pretrained("BAAI/bge-m3").eval()
        tk = tr.AutoTokenizer.from_pretrained("BAAI/bge-m3")

        @torch.no_grad()
        def m3(texts):
            out = []
            for i in range(0, len(texts), 32):
                e = tk(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
                h = m(**e).last_hidden_state[:, 0]
                out.append(h / h.norm(dim=-1, keepdim=True))
            return torch.cat(out)

        D = m3(paras)
        sc = {s: m3([q for q, _ in sets[s]]) @ D.T for s in sets}
        results["bge-m3 (568 М, предобучена)"] = report("bge-m3", sc, sets)
        results["гибрид 0.5 · bge-m3 + 0.5 · TF-IDF"] = report(
            "гибрид 0.5 · bge-m3 + 0.5 · TF-IDF", {s: 0.5 * sc[s] + 0.5 * tfidf_scores[s] for s in sets}, sets)

    if args.record:
        args.record.write_text(json.dumps(record_viz(corpus, snaps, probe, sets, results), ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"записано: {args.record}")


def record_viz(corpus: dict, snaps: list[dict], probe: list[int], sets: dict, results: dict) -> dict:
    """Данные для V-embedding-space: проекция PCA до и после, косинусы «своих» и случайных пар, кривые."""
    def pca2(D):
        X = D[probe] - D[probe].mean(0)
        _, _, V = torch.pca_lowrank(X, q=2, center=False)
        P = X @ V[:, :2]
        return (P / P.abs().max()).tolist()

    def rand_cos(D, seed=0):
        g = torch.Generator().manual_seed(seed)
        i, j = torch.randint(len(D), (2, 600), generator=g)
        return [round(float(x), 4) for x in (D[i] * D[j]).sum(-1)[i != j]]

    tasks = [corpus["paragraphs"][i]["task"] for i in probe]
    first, last = snaps[0], snaps[-1]
    return {
        "probe": [{"task": t, "block": t[0], "text": corpus["paragraphs"][i]["text"][:140]} for t, i in zip(tasks, probe)],
        "before": {"xy": pca2(first["D"]), "pos": [round(x, 4) for x in first["pos"]], "rand": rand_cos(first["D"])},
        "after": {"xy": pca2(last["D"]), "pos": [round(x, 4) for x in last["pos"]], "rand": rand_cos(last["D"])},
        "curve": [{"step": s["step"], "loss": s.get("loss"), "r_frag": s["r_frag"], "r_q": s["r_q"],
                   "aniso": round(s["aniso"], 4)} for s in snaps],
        "results": {name: {s: {str(k): round(v, 4) for k, v in r.items()} for s, r in row.items()}
                    for name, row in results.items()},
    }


if __name__ == "__main__":
    main()
