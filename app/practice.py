"""Reading practice: compare what a learner read aloud with the sentence on the screen.

Both texts are put in the form m3 was trained on (lowercase, no punctuation, digits spelled out), with accents
and hyphens dropped and short pronouns read as full ones, so è/e, pitit-la/pitit la or w/ou never decides a
word. Machine hearing is not perfect: a word marked as missed can be the model's mistake, and the page says so.
Recordings are not kept.
"""
import difflib

import asr_normalize
import engine

# The sentences come from the examples already used in the Space, grouped by who they are for.
LESSONS = [
    {"id": "bonjou", "title": "Bonjou", "en": "Everyday phrases", "sentences": [
        "Bonjou! Mwen kontan anpil pale avè w jodi a.",
        "Kijan ou ye?",
        "Mwen espere tout bagay ap mache byen pou ou.",
        "Si w bezwen plis enfòmasyon, tanpri rele biwo nou an oswa voye yon mesaj.",
        "Pa bliye bwè dlo ki pwòp epi lave men w ak savon.",
    ]},
    {"id": "timoun", "title": "Pou timoun", "en": "For children", "sentences": [
        "Te gen yon ti lapen ki te renmen kouri nan jaden an chak maten.",
        "Yon jou, li jwenn yon gwo kawòt!",
        "Pitit yo, lave men nou byen ak savon anvan nou manje, tande!",
        "Manman, gade desen mwen fè lekòl la!",
        "Se yon kay ak yon gwo solèy jòn.",
        "Pitit la te kontan anpil lè manman l pote yon bèl gato pou fèt li.",
    ]},
    {"id": "adolesan", "title": "Pou adolesan", "en": "For teenagers", "sentences": [
        "Mezanmi, egzamen matematik la te difisil anpil, men mwen kwè m te reyisi l!",
        "Frè m, ou tande nouvo mizik la?",
        "Tout moun lekòl la ap chante l depi maten.",
        "Papa, èske nou ka al jwe foutbòl nan lakou a apre m fin fè devwa m?",
        "Lekòl la ap louvri pòt li lendi pwochen a 8:30 nan maten.",
    ]},
    {"id": "granmoun", "title": "Pou granmoun", "en": "For adults", "sentences": [
        "Bonjou, mwen rele pou konfime randevou mwen demen a dizè nan maten.",
        "Si w vle louvri yon kont nan bank lan, pote kat idantite w ak yon prèv adrès.",
        "Nou dwe fini pwojè a anvan vandredi, kidonk ann travay ansanm pou n rive fè l.",
        "Nou dwe pwoteje anviwònman an pou jenerasyon k ap vini yo.",
        "Lapli a tonbe tout lannwit, men solèy la leve bèl bonè maten an.",
    ]},
    {"id": "granmoun-aje", "title": "Pou granmoun aje", "en": "For elders", "sentences": [
        "Pitit mwen, lè m te jèn, nou te konn chita anba pye mango a pou n tande istwa granmoun yo.",
        "Tande m byen, pitit gason m: piti piti zwazo fè nich li.",
        "Travay ak pasyans, w ap rive.",
    ]},
]


# The short and full pronouns are the same word read either way (avè w / avè ou), and the model often writes
# the full form for the short one, so neither decides a word.
FULL_FORM = {"m": "mwen", "w": "ou", "l": "li", "n": "nou", "y": "yo", "k": "ki"}


def words_of(text):
    return [FULL_FORM.get(w, w) for w in asr_normalize.metric_form(text).replace("-", " ").split()]


def align(ref, hyp):
    """Levenshtein word alignment: pairs of (ref index or None, hyp index or None), in order."""
    n, m = len(ref), len(hyp)
    d = [[i + j if i * j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def compare(expected, heard):
    """Mark each word on the screen: ok (heard), close (nearly: one or two letters off), miss."""
    tokens = expected.split()
    ref, owner = [], []
    for t, token in enumerate(tokens):
        for w in words_of(token):
            ref.append(w)
            owner.append(t)
    hyp = words_of(heard)
    status, extra = ["miss"] * len(ref), []
    for ri, hi in align(ref, hyp):
        if ri is None:
            extra.append(hyp[hi])
        elif hi is not None:
            if ref[ri] == hyp[hi]:
                status[ri] = "ok"
            elif difflib.SequenceMatcher(None, ref[ri], hyp[hi]).ratio() >= 0.75:
                status[ri] = "close"
    rank = {"ok": 0, "close": 1, "miss": 2}
    marks = ["skip"] * len(tokens)            # a token with no letters (a dash) is not judged
    for ri, t in enumerate(owner):
        if marks[t] == "skip" or rank[status[ri]] > rank[marks[t]]:
            marks[t] = status[ri]
    return {"tokens": [{"text": tok, "status": st} for tok, st in zip(tokens, marks)],
            "ok": status.count("ok"), "close": status.count("close"), "total": len(ref),
            "score": round(100 * status.count("ok") / len(ref)) if ref else 0, "extra": extra}


def check(expected, audio):
    expected = " ".join((expected or "").split())
    if not expected:
        raise engine.UserError("Pa gen fraz pou konpare. (There is no sentence to compare with.)")
    heard = engine.transcribe_bytes(audio, max_s=120)["text"]
    return {**compare(expected, heard), "heard": heard}
