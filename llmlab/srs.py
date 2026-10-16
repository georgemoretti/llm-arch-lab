"""Интервальное повторение вопросов (SM-2, вкладка Interview).

Самооценка ответа — одна из четырёх кнопок, они переводятся в оценку q алгоритма SM-2:

| кнопка | q | что происходит |
|--------|---|----------------|
| again  | 2 | провал: повтор через 10 минут, серия сбрасывается, lapses + 1 |
| hard   | 3 | вспомнил с трудом |
| good   | 4 | вспомнил |
| easy   | 5 | легко; интервал дополнительно × 1.3 |

Лёгкость (ease) меняется по формуле SM-2: EF += 0.1 − (5 − q)·(0.08 + (5 − q)·0.02), но не ниже 1.3.
Интервалы успешных повторений: 1 день, 6 дней, затем предыдущий × EF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

GRADES = {"again": 2, "hard": 3, "good": 4, "easy": 5}
START_EASE = 2.5
MIN_EASE = 1.3
EASY_BONUS = 1.3
RELEARN_MINUTES = 10


@dataclass
class Card:
    qid: str
    ease: float = START_EASE
    interval_days: float = 0.0
    reps: int = 0                   # успешных повторений подряд
    lapses: int = 0                 # сколько раз забыл
    due_at: str | None = None       # ISO-8601 UTC; None — новая карточка
    last_grade: str | None = None
    last_review_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def schedule(card: Card, grade: str, now: datetime | None = None) -> Card:
    """Новое состояние карточки после ответа с оценкой ``grade`` (again | hard | good | easy)."""
    if grade not in GRADES:
        raise ValueError(f"оценка должна быть одной из {', '.join(GRADES)}")
    now = now or datetime.now(timezone.utc)
    q = GRADES[grade]
    ease = max(MIN_EASE, card.ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if q < 3:
        reps, lapses, interval = 0, card.lapses + 1, 0.0
        due = now + timedelta(minutes=RELEARN_MINUTES)
    else:
        reps, lapses = card.reps + 1, card.lapses
        interval = 1.0 if reps == 1 else 6.0 if reps == 2 else round(card.interval_days * ease, 1)
        if grade == "easy":
            interval = round(interval * EASY_BONUS, 1)
        due = now + timedelta(days=interval)
    return Card(card.qid, round(ease, 3), interval, reps, lapses, _iso(due), grade, _iso(now))


def preview(card: Card, now: datetime | None = None) -> dict[str, float]:
    """Через сколько дней карточка вернётся при каждой из оценок (для подписей на кнопках)."""
    now = now or datetime.now(timezone.utc)
    out = {}
    for g in GRADES:
        nxt = schedule(card, g, now)
        out[g] = round((datetime.fromisoformat(nxt.due_at) - now).total_seconds() / 86400, 3)
    return out


def is_due(card: Card | None, now: datetime | None = None) -> bool:
    if card is None or card.due_at is None:
        return True
    return datetime.fromisoformat(card.due_at) <= (now or datetime.now(timezone.utc))
