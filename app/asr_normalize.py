# Copy of scripts/whisper_ft/normalize.py (the text form m3 was trained and scored on); keep the two in sync.
"""One canonical spelling for every dataset, shared by training and scoring.

The sources disagree: the Bible text is truecase with punctuation, the radio transcripts are lowercase
with none, `history-data` is lowercase with glued clitics. A model trained on the mix has to guess which
convention a given recording wants, and every wrong guess counts as an error. So the training targets and
the metric both go through `normalize`, and the model only ever has to learn one form: lowercase, no
punctuation, digits spelled out, modern spelling without apostrophes.

Casing and punctuation are a separate job for a text model on top of the transcript.
"""
import re
import unicodedata

# Kreyòl numerals. 0-19 are irregular; 20-99 is built below; the forms follow common usage
# (25 vennsenk, 75 swasannkenz, 98 katrevendizuit, 2026 de mil vennsis).
UNITS = ["zewo", "en", "de", "twa", "kat", "senk", "sis", "sèt", "uit", "nèf",
         "dis", "onz", "douz", "trèz", "katòz", "kenz", "sèz", "disèt", "dizuit", "diznèf"]
TENS = {20: "ven", 30: "trant", 40: "karant", 50: "senkant", 60: "swasant"}
NASAL = {20: "venn", 30: "trann", 40: "karann", 50: "senkann", 60: "swasann"}
HARD = {20: "vent", 30: "trant", 40: "karant", 50: "senkant", 60: "swasant"}
HUNDREDS = [None, "san", "desan", "twasan", "katsan", "senksan", "sisan", "sètsan", "uisan", "nèfsan"]

LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# Tokens that appear in French but not in Kreyòl, used to flag French-language rows
FRENCH = {"les", "des", "est", "sont", "nous", "vous", "dans", "avec", "qui", "que", "qu", "une",
          "cette", "ces", "il", "elle", "ils", "elles", "leur", "notre", "votre", "etait", "était",
          "peut", "faire", "aussi", "donc", "alors", "tres", "très", "beaucoup", "toujours", "jamais",
          "français", "francais", "haïti", "sur", "plus", "cet", "vers", "chez", "quelque", "le",
          "du", "au", "aux", "ce", "vient", "suis", "sais", "veux", "avez", "avons", "ont", "été",
          "comment", "pourquoi", "aujourd", "hui", "parce", "puis", "ensuite", "toute", "autre",
          "meme", "même", "c", "j", "quand", "notamment", "ainsi"}
# Kreyòl writes only à, è and ò; any other accent is a French (or English) spelling
NON_KREYOL = re.compile(r"[éêëçùûîïôœæáíúñ]")


def spell_int(n):
    """Spell a non-negative integer below one million."""
    if n < 20:
        return UNITS[n]
    if n < 100:
        ten, unit = n // 10 * 10, n % 10
        if ten == 70:
            return "swasann" + UNITS[10 + unit]
        if ten == 80:
            return "katreven" if unit == 0 else f"katreven {UNITS[unit]}"
        if ten == 90:
            return "katreven" + UNITS[10 + unit]
        if unit == 0:
            return TENS[ten]
        if unit == 1:
            return HARD[ten] + "eyen"
        if unit in (8, 9):
            return HARD[ten] + UNITS[unit]
        return NASAL[ten] + UNITS[unit]
    if n < 1000:
        hund, rest = n // 100, n % 100
        return HUNDREDS[hund] + ("" if rest == 0 else " " + spell_int(rest))
    thou, rest = n // 1000, n % 1000
    head = "mil" if thou == 1 else f"{spell_int(thou)} mil"
    return head + ("" if rest == 0 else " " + spell_int(rest))


def spell_digits(text):
    """Replace every run of digits with its Kreyòl words, plus the symbols that read as words."""
    text = re.sub(r"(\d)\s*%", r"\1 pousan", text)
    text = re.sub(r"(?<=\d)[  ,](?=\d{3}\b)", "", text)          # 1 000 / 1,000 -> 1000

    def one(m):
        whole, frac = m.group(1), m.group(2)
        n = int(whole)
        out = spell_int(n) if n < 1_000_000 else " ".join(spell_int(int(c)) for c in whole)
        if frac:
            out += " pwen " + " ".join(spell_int(int(c)) for c in frac)
        return out

    return re.sub(r"(\d+)(?:[.,](\d+))?", one, text)


def modern_spelling(text):
    """Drop the apostrophes the older orthography used: `pa t' gen` -> `pa t gen`, `t'ap` -> `t ap`."""
    t = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    t = re.sub(r"(\w)'(?=[\s,.;:!?\"»”)\]]|$)", r"\1", t)
    t = re.sub(r"(\w)'(?=\w)", r"\1 ", t)
    return t


def normalize(text, lower=True, punct=False):
    """The canonical form used for both training targets and error rates."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("​", " ").replace("\xa0", " ")
    t = re.sub(r"[“”«»]", '"', t).replace("–", "-").replace("—", "-").replace("…", " ")
    t = modern_spelling(t)
    t = spell_digits(t)
    if lower:
        t = t.lower()
    if not punct:
        t = re.sub(r"[^\w\s'-]", " ", t, flags=re.UNICODE)
        t = re.sub(r"(?<!\w)[-']|[-'](?!\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def metric_form(text):
    """Strip accents on top of `normalize`, so è/e and ò/o never decide a word error."""
    t = unicodedata.normalize("NFKD", normalize(text))
    return "".join(c for c in t if not unicodedata.combining(c))


def french_score(text):
    """Share of tokens that are French-only words; radio broadcasts switch into French mid-sentence."""
    toks = [p for t in normalize(text).split() for p in t.split("-") if p]
    if not toks:
        return 0.0
    return sum(t in FRENCH for t in toks) / len(toks)


def is_french(text, min_share=0.18):
    """A row written in French rather than Kreyòl.

    Kreyòl prose about Haiti is full of French-spelled proper nouns (Éstimé, François, Déjoie), so the
    accent test skips capitalised words and only judges ordinary ones.

    Catches French *text*. It cannot catch French *speech* that a Kreyòl-only ASR wrote down as
    Kreyòl-looking words, which is how French appears in the radio transcripts.
    """
    ordinary = []
    for word in text.split():
        first = next((c for c in word if c.isalpha()), "")
        if not first or first.islower():
            ordinary.append(word)
    return french_score(text) >= min_share or len(NON_KREYOL.findall(" ".join(ordinary))) >= 2


def suspect(text, duration=None):
    """Reasons a row is bad training data. Empty list means the row looks usable."""
    why, norm = [], normalize(text)
    if len(norm) < 2:
        why.append("empty")
    if re.search(r"\d", norm):
        why.append("digits_left")
    if is_french(text):
        why.append("french")
    letters = len(LETTER.findall(norm))
    if letters and len(norm) and letters / len(norm) < 0.7:
        why.append("low_letter_ratio")
    if norm and max((len(w) for w in norm.split()), default=0) > 24:
        why.append("long_token")
    if duration:
        rate = letters / duration
        if rate < 5 or rate > 25:
            why.append(f"rate_{rate:.0f}")
    return why


if __name__ == "__main__":
    for n in (0, 1, 8, 15, 21, 25, 31, 48, 70, 75, 80, 81, 90, 98, 100, 101, 250, 1000, 1804, 2026, 15000):
        print(f"{n:>6} {spell_int(n)}")
    for s in ["Jenèz chapit pemye. Istwa kreyasyon an.",
              "Nan lane 1804, 25 moun te peye 50% nan 2,500 goud.",
              "pa t' gen anyen, epi t'ap vini",
              "Qu'est-ce que le vaudou, d'où vient-il?"]:
        print(f"\n{s}\n-> {normalize(s)}\n   french={french_score(s):.2f} suspect={suspect(s)}")
