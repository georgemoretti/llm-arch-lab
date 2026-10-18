"""BPE: кодирование строки по рангам слияний и декодирование через байты.

Реализуй encode и decode. merges и vocab — как в A11: {(a, b): 256 + k} и {id: bytes}.
"""

from __future__ import annotations

from llmlab.core.bpe import merge_pair   # готовое слияние пары (эталон задачи «BPE: обучение»)


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Строка → id: байты UTF-8, затем слияния по рангу — каждый раз пара с наименьшим id из merges."""
    # TODO 1: начни с байтов: list(text.encode("utf-8")).
    # TODO 2: пока в последовательности есть хотя бы одна пара из merges:
    #         среди соседних пар найди ту, что выучена РАНЬШЕ всех (наименьший id в merges),
    #         и слей её везде через merge_pair. Частота пары в этом тексте здесь не важна — почему?
    # TODO 3: пар из merges не осталось → готово. Не забудь про строки из 0 и 1 байта.
    raise NotImplementedError("encode")


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """id → строка: сначала склеить байты всех токенов, потом один раз декодировать UTF-8."""
    # TODO: токен может быть половиной символа (например, b" \xd0"). Как собрать строку так, чтобы символ,
    #       разрезанный между двумя токенами, не превратился в «�»? Неполные байты в конце не должны
    #       ронять decode: посмотри на параметр errors у bytes.decode.
    raise NotImplementedError("decode")
