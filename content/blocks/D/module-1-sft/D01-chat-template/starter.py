"""Chat-шаблон: список ходов диалога → строка со служебными токенами ролей → id.

Реализуй три функции. Правила шаблона — как у Qwen3-0.6B (без tools); подробно — в README.
"""

from __future__ import annotations

import re


def render_chat(messages: list[dict], add_generation_prompt: bool = False, enable_thinking: bool = True) -> str:
    """Список ходов ``{"role": ..., "content": ...}`` → строка chat-шаблона Qwen3.

    Каждый ход — ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. У ассистента рассуждение ``<think>…</think>``
    вырезается из ходов ДО последнего вопроса пользователя; последний ход ассистента после этого вопроса получает
    блок ``<think>\\n{рассуждение}\\n</think>\\n\\n`` (пустой, если рассуждения нет).
    """
    # TODO 1: найди индекс последнего хода с role == "user" (если такого нет — len(messages) - 1).
    # TODO 2: system и user — '<|im_start|>{role}\n{content}<|im_end|>\n'. Неизвестная роль — ValueError.
    # TODO 3: assistant. Рассуждение — поле "reasoning_content", если это строка; иначе, если в content есть
    #         '</think>', рассуждение — текст между '<think>' и '</think>' (без крайних '\n'), а content — то, что после
    #         '</think>' (без ведущих '\n').
    # TODO 4: если ход стоит ПОСЛЕ последнего вопроса и (это последний ход ИЛИ рассуждение непустое) —
    #         '<|im_start|>assistant\n<think>\n{рассуждение}\n</think>\n\n{content}', иначе '<|im_start|>assistant\n{content}';
    #         в обоих случаях затем '<|im_end|>\n'.
    # TODO 5: add_generation_prompt → '<|im_start|>assistant\n' (открытый ход), а при enable_thinking=False ещё
    #         '<think>\n\n</think>\n\n'.
    raise NotImplementedError("render_chat")


def split_special(text: str, special_tokens: list[str]) -> list[str]:
    """Разрезать строку по служебным токенам: служебный токен — отдельный кусок, пустых кусков нет."""
    # TODO: re.split с группой (скобками) сохраняет разделители. Экранируй токены re.escape; длинные — раньше
    #       коротких. Выкинь пустые строки.
    raise NotImplementedError("split_special")


def encode_chat(text: str, tok) -> list[int]:
    """Строка шаблона → id: служебный токен — один id (``tok.token_to_id``), текст между ними — ``tok.encode``.

    ``tok`` — объект с полями ``special_tokens`` (list[str]) и методами ``token_to_id(str) -> int``,
    ``encode(str) -> list[int]`` (обычный текст, служебные токены НЕ разбирает).
    """
    # TODO: пройди по кускам split_special(text, tok.special_tokens); служебный кусок → [tok.token_to_id(кусок)],
    #       обычный → tok.encode(кусок). Склей в один list[int].
    raise NotImplementedError("encode_chat")
