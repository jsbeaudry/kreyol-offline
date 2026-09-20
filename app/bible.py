"""The Kreyòl Bible, for reading along and listening.

The text lives in app/data/bible.json, one sentence per line, built by app/make_bible.py from the Bible
pipeline's own files. It is not part of the kit: the page shows the Bib tab only when the file is there,
so nobody has to ship the text who does not already have it.
"""
import json
import os
import threading

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bible.json")
_lock = threading.Lock()
_loaded = None


def available():
    return os.path.exists(DATA)


def _load():
    global _loaded
    with _lock:
        if _loaded is None:
            with open(DATA, encoding="utf-8") as f:
                raw = json.load(f)
            _loaded = (raw["books"], {(c["book"], c["chapter"]): c["lines"] for c in raw["chapters"]})
    return _loaded


def books():
    return _load()[0]


def chapter(book, number):
    """One chapter as its lines, or None if there is no such book or chapter."""
    all_books, chapters = _load()
    lines = chapters.get((book, number))
    if lines is None:
        return None
    name = next((b["name"] for b in all_books if b["id"] == book), book)
    return {"book": book, "name": name, "chapter": number, "lines": lines,
            "chars": sum(len(line) for line in lines)}
