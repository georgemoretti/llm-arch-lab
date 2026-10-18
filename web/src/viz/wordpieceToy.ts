// Игрушечный WordPiece для визуализаций V-wordpiece и V-mlm-masking: та же логика, что в B07 (core/bert.py),
// и маленький словарь из частых кусков английского и русского. Все буквы есть и как начало, и как «##буква»,
// поэтому [UNK] получают только слова с символами вне алфавитов — как в настоящем словаре BERT.

const WHOLE = (
  "the a is are of and to in on for with you all need it this that " +
  "un una play token trans attention bert model learn pre train mask word piece " +
  "и в на не что это как по для модель токен внима обуч слов текст мир при пред сеть " +
  "учится восстанавливать закрыт слова контексту обеих сторон с"
).split(" ");
const CONT = (
  "aff ff able ab le ing ed er ers s ization izer ize form former formers ion ions ly ness al " +
  "ель ели и ы а у ом ов ение ния ние ть ет ают ка ки вет ные ный ная ое ий ая " +
  "слов ов ам ами атор ия иза ация ая обуч тора ые"
).split(" ");
const LETTERS = "abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя0123456789".split("");

export const TOY_VOCAB: Set<string> = new Set([
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
  ...WHOLE, ...LETTERS,
  ...CONT.map((c) => "##" + c), ...LETTERS.map((c) => "##" + c),
  ".", ",", "!", "?", "—", "-", ":", ";", "(", ")", "«", "»", "'", '"',
]);

export const isPunct = (ch: string) => {
  const cp = ch.codePointAt(0)!;
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(ch);
};

export function normalize(text: string, lower: boolean): string {
  // eslint-disable-next-line no-control-regex
  let t = text.replace(/[\u0000�]/g, "").replace(/[\t\n\r\p{Zs}]/gu, " ").replace(/[\p{Cc}\p{Cf}]/gu, "");
  if (lower) t = t.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
  return t;
}

export function basicTokenize(text: string, lower = true): string[] {
  const words: string[] = [];
  for (const chunk of normalize(text, lower).split(" ").filter(Boolean)) {
    let cur = "";
    for (const ch of Array.from(chunk)) {
      if (isPunct(ch)) {
        if (cur) words.push(cur);
        words.push(ch);
        cur = "";
      } else cur += ch;
    }
    if (cur) words.push(cur);
  }
  return words;
}

export interface Attempt {
  piece: string;
  found: boolean;
}

/** Жадный разбор слова; steps — все проверенные кандидаты по порядку (для показа по шагам). */
export function wordpiece(word: string, vocab: Set<string> = TOY_VOCAB): { pieces: string[]; steps: Attempt[][] } {
  const chars = Array.from(word);
  const pieces: string[] = [];
  const steps: Attempt[][] = [];
  let start = 0;
  while (start < chars.length) {
    const tries: Attempt[] = [];
    let end = chars.length;
    let found = "";
    while (end > start) {
      const piece = (start === 0 ? "" : "##") + chars.slice(start, end).join("");
      const ok = vocab.has(piece);
      tries.push({ piece, found: ok });
      if (ok) {
        found = piece;
        break;
      }
      end--;
    }
    steps.push(tries);
    if (!found) return { pieces: ["[UNK]"], steps };
    pieces.push(found);
    start = end;
  }
  return { pieces, steps };
}

/** Текст → токены с номером слова (−1 у спецтокенов не бывает: их добавляет вызывающий). */
export function tokenizeWithWords(text: string, lower = true): { tokens: string[]; wordIds: number[] } {
  const tokens: string[] = [];
  const wordIds: number[] = [];
  basicTokenize(text, lower).forEach((w, i) => {
    for (const p of wordpiece(w).pieces) {
      tokens.push(p);
      wordIds.push(i);
    }
  });
  return { tokens, wordIds };
}
