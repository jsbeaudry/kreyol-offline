"""Build app/data/bible.json for the Bib tab, from the two files the Bible pipeline produced.

The chapter text comes from `bible.json` (one sentence per line, which is what the page reads line by
line); the book names come from `merged_data.json`, because in `bible.json` all but the first three books
are codes (`nbr`, `7pr`, `mmN`). Both list the same 1,189 chapters in the same order, and this checks that
before trusting it.

    python3 app/make_bible.py ~/path/bible.json ~/path/merged_data.json

The Bible text itself is not kept in git: the page simply shows the Bib tab when app/data/bible.json is
there, and leaves it out when it is not.
"""
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "bible.json")


def slug(name):
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    text_file, names_file = sys.argv[1], sys.argv[2]
    with open(text_file, encoding="utf-8") as f:
        chapters = json.load(f)
    with open(names_file, encoding="utf-8") as f:
        named = json.load(f)
    if len(chapters) != len(named):
        sys.exit(f"the two files disagree: {len(chapters)} chapters vs {len(named)}")
    mismatch = [i for i, (a, b) in enumerate(zip(chapters, named)) if a["chapter"] != b["chapter"]]
    if mismatch:
        sys.exit(f"chapter numbers differ at {len(mismatch)} places, starting at {mismatch[0]}")

    books, out = [], []
    for text_ch, named_ch in zip(chapters, named):
        name = named_ch["identifier"].strip()
        if not books or books[-1]["name"] != name:
            books.append({"id": slug(name), "name": name, "chapters": 0})
        books[-1]["chapters"] += 1
        lines = [" ".join(line.split()) for line in text_ch["text"].split("\n")]
        out.append({"book": books[-1]["id"], "chapter": text_ch["chapter"],
                    "lines": [line for line in lines if line]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"books": books, "chapters": out}, f, ensure_ascii=False)
    lines = sum(len(c["lines"]) for c in out)
    chars = sum(len(line) for c in out for line in c["lines"])
    print(f"{OUT}: {len(books)} books, {len(out)} chapters, {lines} lines, {chars/1e6:.1f}M characters")
    print("books:", ", ".join(f"{b['name']} {b['chapters']}" for b in books[:6]), "...")


if __name__ == "__main__":
    main()
