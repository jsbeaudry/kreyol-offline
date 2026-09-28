"""Kreyòl text normalization for TTS, matching the training data: spell out numbers, split long text into chunks."""
import re
import unicodedata

UNITS = ["zewo", "en", "de", "twa", "kat", "senk", "sis", "sèt", "uit", "nèf",
         "dis", "onz", "douz", "trèz", "katòz", "kenz", "sèz", "disèt", "dizuit", "diznèf"]
TENS = {2: "ven", 3: "trant", 4: "karant", 5: "senkant", 6: "swasant"}
# Tens end in -n before de/twa/kat/senk/sis (vennde, karannsenk) and in -t otherwise (venteyen, ventsèt).
N_STEM = {2: "venn", 3: "trann", 4: "karann", 5: "senkann", 6: "swasann"}
T_STEM = {2: "vent", 3: "trant", 4: "karant", 5: "senkant", 6: "swasant"}
HUNDREDS = {1: "san", 2: "de san", 3: "twa san", 4: "kat san", 5: "sen san",
            6: "si san", 7: "sèt san", 8: "ui san", 9: "nèf san"}
ORDINALS = {1: "premye", 2: "dezyèm", 3: "twazyèm", 4: "katriyèm", 5: "senkyèm",
            6: "sizyèm", 7: "setyèm", 8: "uityèm", 9: "nevyèm", 10: "dizyèm"}


def _below_100(n):
    if n < 20:
        return UNITS[n]
    tens, unit = divmod(n, 10)
    if tens == 7:
        return "swasantonz" if unit == 1 else "swasann" + UNITS[10 + unit]
    if tens == 9:
        return "katreven" + UNITS[10 + unit]
    if tens == 8:
        if unit == 0:
            return "katreven"
        return "katreven en" if unit == 1 else "katreven" + UNITS[unit]
    if unit == 0:
        return TENS[tens]
    if unit == 1:
        return T_STEM[tens] + "eyen"
    if unit >= 7:
        return T_STEM[tens] + UNITS[unit]
    return N_STEM[tens] + UNITS[unit]


def int_to_kreyol(n):
    if n == 0:
        return "zewo"
    parts = []
    for size, one, many in ((10**9, "yon milya", "milya"), (10**6, "yon milyon", "milyon"), (1000, "mil", "mil")):
        count, n = divmod(n, size)
        if count:
            parts.append(one if count == 1 else f"{int_to_kreyol(count)} {many}")
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append(HUNDREDS[hundreds])
    if rest:
        parts.append(_below_100(rest))
    return " ".join(parts)


def _number(token):
    m = re.fullmatch(r"(\d+)([.,])(\d+)", token)
    if not m:
        return int_to_kreyol(int(token))
    whole, sep, frac = m.groups()
    frac_words = " ".join(UNITS[int(d)] for d in frac) if frac.startswith("0") else int_to_kreyol(int(frac))
    return f"{int_to_kreyol(int(whole))} {'vigil' if sep == ',' else 'pwen'} {frac_words}"


def spell_numbers(text):
    t = re.sub(r"(?<=\d)[ ,.](?=\d{3}\b)", "", text)  # 2,500 / 2.500 / 2 500 → 2500
    t = re.sub(r"\b(\d{1,2})[:h](\d{2})\b",
               lambda m: f"{int_to_kreyol(int(m.group(1)))} è"
               + ("" if m.group(2) == "00" else f" {int_to_kreyol(int(m.group(2)))}"), t)
    t = re.sub(r"\$\s*(\d+(?:[.,]\d+)?)", lambda m: f"{_number(m.group(1))} dola", t)
    t = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", lambda m: f"{_number(m.group(1))} pousan", t)
    t = re.sub(r"\b(\d+)(?:yèm|yem|ème|eme|èm|ye|er|re|e)\b",
               lambda m: ORDINALS.get(int(m.group(1))) or int_to_kreyol(int(m.group(1))) + "yèm", t)
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1 sou \2", t)
    t = re.sub(r"\d+(?:[.,]\d+)?", lambda m: _number(m.group(0)), t)
    return t


QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "«": '"', "»": '"'})


def normalize(text):
    t = unicodedata.normalize("NFC", text or "").translate(QUOTES)
    return spell_numbers(re.sub(r"\s+", " ", t).strip())


def split_text(text, max_chars=220):
    """Group sentences into chunks of at most max_chars (the model was trained on clips of 15 s or less)."""
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    pieces = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            cut = chunk.rfind(",", 0, max_chars)
            if cut < max_chars // 2:
                cut = chunk.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(chunk[: cut + 1].strip())
            chunk = chunk[cut + 1 :].strip()
        if chunk:
            pieces.append(chunk)
    return pieces
