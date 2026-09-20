"""The settings the page can change, kept in travay/settings.json.

The defaults come from a measurement on five minutes of speech from a YouTube recording: with lines merged
across pauses of up to 2 s and cut at 28 s, 7.5% of the words were inside a repeated run (the model looping
on a long line, one line 31% repetition). With a 0.8 s gap and a 20 s cap, none were, and the same speech
came out 100 words shorter because the loops were gone.
"""
import json
import os
import threading

FIELDS = {
    # name: (default, low, high, what it does)
    "vad_threshold": (0.5, 0.1, 0.9, "How sure the detector must be that it hears speech. Higher finds more silence."),
    "vad_min_silence_ms": (100, 20, 3000, "How long a pause must be before it counts as silence at all."),
    "merge_gap_s": (0.8, 0.0, 3.0, "Pauses shorter than this stay inside a line instead of starting a new one."),
    "max_region_s": (20.0, 5.0, 30.0, "The longest a line may be. Long lines make the model repeat itself."),
    "pad_s": (0.2, 0.0, 1.0, "Extra audio kept on each side of a line, so words are not clipped."),
    "sentence_pause_s": (0.25, 0.0, 2.0, "Silence between sentences when the voice reads a document."),
    "paragraph_pause_s": (0.8, 0.0, 4.0, "Silence between paragraphs when the voice reads a document."),
}
PATH = None          # set by server.main() to travay/settings.json
_lock = threading.RLock()        # all() is called while update() holds it
_values = None


def defaults():
    return {name: spec[0] for name, spec in FIELDS.items()}


def describe():
    return [{"name": name, "default": spec[0], "min": spec[1], "max": spec[2], "about": spec[3],
             "step": 0.05 if isinstance(spec[0], float) and spec[2] <= 3 else (0.1 if isinstance(spec[0], float) else 10)}
            for name, spec in FIELDS.items()]


def clean(changes):
    """Keep what is known and in range, as the right kind of number."""
    out = {}
    for name, value in (changes or {}).items():
        if name not in FIELDS:
            continue
        default, low, high, _ = FIELDS[name]
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        number = min(high, max(low, number))
        out[name] = number if isinstance(default, float) else int(round(number))
    return out


def all():
    global _values
    with _lock:
        if _values is None:
            _values = defaults()
            if PATH and os.path.exists(PATH):
                try:
                    with open(PATH, encoding="utf-8") as f:
                        _values.update(clean(json.load(f)))
                except ValueError:
                    pass
        return dict(_values)


def get(name):
    return all()[name]


def update(changes):
    global _values
    kept = clean(changes)
    with _lock:
        _values = {**all(), **kept}
        if PATH:
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            tmp = PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_values, f, indent=1)
            os.replace(tmp, PATH)
        return dict(_values)


def reset():
    return update(defaults())
