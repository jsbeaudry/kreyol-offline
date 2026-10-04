/**
 * Text as the Kreyòl voice was trained to read it — a port of kreyol-tts/kreyol_text.py and Klara's
 * tts_text.speakable(), plus the character rules from the fine-tune's Pocket config. Checked against
 * the Python on the same inputs (see the export folder's text check).
 */
const UNITS = ['zewo', 'en', 'de', 'twa', 'kat', 'senk', 'sis', 'sèt', 'uit', 'nèf',
  'dis', 'onz', 'douz', 'trèz', 'katòz', 'kenz', 'sèz', 'disèt', 'dizuit', 'diznèf'];
const TENS: Record<number, string> = { 2: 'ven', 3: 'trant', 4: 'karant', 5: 'senkant', 6: 'swasant' };
// Tens end in -n before de/twa/kat/senk/sis (vennde, karannsenk) and in -t otherwise (venteyen, ventsèt).
const N_STEM: Record<number, string> = { 2: 'venn', 3: 'trann', 4: 'karann', 5: 'senkann', 6: 'swasann' };
const T_STEM: Record<number, string> = { 2: 'vent', 3: 'trant', 4: 'karant', 5: 'senkant', 6: 'swasant' };
const HUNDREDS: Record<number, string> = { 1: 'san', 2: 'de san', 3: 'twa san', 4: 'kat san', 5: 'sen san',
  6: 'si san', 7: 'sèt san', 8: 'ui san', 9: 'nèf san' };
const ORDINALS: Record<number, string> = { 1: 'premye', 2: 'dezyèm', 3: 'twazyèm', 4: 'katriyèm', 5: 'senkyèm',
  6: 'sizyèm', 7: 'setyèm', 8: 'uityèm', 9: 'nevyèm', 10: 'dizyèm' };
// Expanded only right after a number: the "m" in "m ap vini" is a pronoun.
const MEASURES: Record<string, string> = {
  kg: 'kilogram', g: 'gram', mg: 'miligram', lb: 'liv', t: 'tòn',
  km: 'kilomèt', m: 'mèt', cm: 'santimèt', mm: 'milimèt',
  l: 'lit', ml: 'mililit', cl: 'santilit',
  h: 'èdtan', min: 'minit', s: 'segonn',
};
// From the fine-tune's config: characters absent from its training data.
const REPLACE: Record<string, string> = {
  '"': '', '“': '', '”': '', '„': '', '«': '', '»': '', '’': "'", '‘': "'",
  ':': ',', '(': '', ')': '', '[': '', ']': '',
};

function below100(n: number): string {
  if (n < 20) return UNITS[n];
  const tens = Math.floor(n / 10), unit = n % 10;
  if (tens === 7) return unit === 1 ? 'swasantonz' : 'swasann' + UNITS[10 + unit];
  if (tens === 9) return 'katreven' + UNITS[10 + unit];
  if (tens === 8) return unit === 0 ? 'katreven' : unit === 1 ? 'katreven en' : 'katreven' + UNITS[unit];
  if (unit === 0) return TENS[tens];
  if (unit === 1) return T_STEM[tens] + 'eyen';
  if (unit >= 7) return T_STEM[tens] + UNITS[unit];
  return N_STEM[tens] + UNITS[unit];
}

export function intToKreyol(n: number): string {
  if (n === 0) return 'zewo';
  const parts: string[] = [];
  for (const [size, one, many] of [[1e9, 'yon milya', 'milya'], [1e6, 'yon milyon', 'milyon'], [1000, 'mil', 'mil']] as const) {
    const count = Math.floor(n / size);
    n = n % size;
    if (count) parts.push(count === 1 ? one : `${intToKreyol(count)} ${many}`);
  }
  const hundreds = Math.floor(n / 100), rest = n % 100;
  if (hundreds) parts.push(HUNDREDS[hundreds]);
  if (rest) parts.push(below100(rest));
  return parts.join(' ');
}

function number(token: string): string {
  const m = /^(\d+)([.,])(\d+)$/.exec(token);
  if (!m) return intToKreyol(parseInt(token, 10));
  const [, whole, sep, frac] = m;
  const fracWords = frac.startsWith('0') ? [...frac].map((d) => UNITS[+d]).join(' ') : intToKreyol(parseInt(frac, 10));
  return `${intToKreyol(parseInt(whole, 10))} ${sep === ',' ? 'vigil' : 'pwen'} ${fracWords}`;
}

export function spellNumbers(text: string): string {
  let t = text.replace(/(\d)[ ,.](?=\d{3}\b)/g, '$1');
  t = t.replace(/\b(\d{1,2})[:h](\d{2})\b/g, (_, h, m) =>
    `${intToKreyol(+h)} è` + (m === '00' ? '' : ` ${intToKreyol(+m)}`));
  t = t.replace(/\$\s*(\d+(?:[.,]\d+)?)/g, (_, n) => `${number(n)} dola`);
  t = t.replace(/(\d+(?:[.,]\d+)?)\s*%/g, (_, n) => `${number(n)} pousan`);
  t = t.replace(/\b(\d+)(?:yèm|yem|ème|eme|èm|ye|er|re|e)(?![\p{L}\p{N}_])/gu, (_, n) =>
    ORDINALS[+n] ?? intToKreyol(+n) + 'yèm');
  t = t.replace(/(\d)\s*\/\s*(\d)/g, '$1 sou $2');
  t = t.replace(/\d+(?:[.,]\d+)?/g, (m) => number(m));
  return t;
}

/** What Klara says, in the form the voice was trained on: units, numbers, spacing. */
export function speakable(text: string): string {
  const units = Object.keys(MEASURES).sort((a, b) => b.length - a.length).join('|');
  let t = (text || '').normalize('NFC').replace(/[‘’]/g, "'").replace(/[“”«»]/g, '"');
  t = t.replace(new RegExp(`(\\d)\\s*(${units})(?![\\p{L}\\p{N}_])`, 'giu'), (_, d, u) => `${d} ${MEASURES[u.toLowerCase()]}`);
  t = spellNumbers(t.replace(/\s+/g, ' ').trim());
  return t.split(/\s+/).filter(Boolean).join(' ');
}

/** Pocket's own preparation of one chunk: its character rules, a capital, a final period. */
export function pocketPrompt(text: string): { text: string; framesAfterEos: number } {
  let t = [...text].map((c) => (c in REPLACE ? REPLACE[c] : c)).join('').replace(/;/g, ',');
  t = t.replace(/[\n\r]/g, ' ').replace(/ {2}/g, ' ').trim();
  const words = t.split(/\s+/).length;
  if (t && t[0] !== t[0].toUpperCase()) t = t[0].toUpperCase() + t.slice(1);
  if (t && /[\p{L}\p{N}]$/u.test(t)) t += '.';
  return { text: t, framesAfterEos: (words <= 4 ? 3 : 1) + 2 };
}
