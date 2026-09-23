"""Which models Dikte uses, kept in dikte/settings.json.

Same shape as app/settings.py: a table of what may be set, a JSON file written atomically, and anything
unknown dropped on the way in. The difference is that these are files on this machine rather than
numbers, so the check is "does it exist, and is it the right kind of file" instead of a range. A setting
pointing at a model that has since been deleted falls back to the default rather than failing at load
time, when the only sign would be a server that will not start.

Drop another .bin beside the speech model, or another .gguf beside the voice, and it appears in the
menu. Nothing is copied or converted here — `convert/` does that, and `setup.sh` fetches the published
ones.
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import models                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'dikte/settings.json')

# The models come from app/models.py, which the page reads too — one table, so the two cannot come
# to disagree about which quantisations exist or which were measured not to work. The speaker is
# Dikte's own: the page picks a voice per reading, while Dikte needs a standing choice.
FIELDS = {name: (models.default(name), spec['title'], models.folder(name), spec['suffix'])
          for name, spec in models.MODELS.items()}
FIELDS['speaker'] = (os.path.join(ROOT, 'kreyol-tts/voices/kreyol_f1.wav'),
                     'Who reads', os.path.join(ROOT, 'kreyol-tts/voices'), '.wav')
NEEDS_RESTART = set(models.RELOADS)
REJECTED = models.REJECTED
REPOS = {name: spec['repo'] for name, spec in models.MODELS.items()}

_lock = threading.RLock()
_values = None


def defaults():
    return {name: spec[0] for name, spec in FIELDS.items()}


def choices(name):
    """Every file on this machine that could go in this setting."""
    if name in models.MODELS:
        return models.on_disk(name)
    import glob
    _, _, folder, suffix = FIELDS[name]
    return sorted(glob.glob(os.path.join(folder, f'*{suffix}')))


def label(path):
    """What to call a model in a menu: its filename without the extension."""
    return os.path.splitext(os.path.basename(path))[0]


def options(name):
    """Everything this setting could be, published or already here."""
    if name in models.MODELS:
        return models.options(name, get(name))
    return [{'path': p, 'label': label(p), 'mb': round(os.path.getsize(p) / 1e6), 'note': '',
             'local': True, 'rejected': None} for p in choices(name)]


def fetch(name, path, progress=None):
    """Download a published file; the page and Dikte share the one implementation."""
    return models.fetch(name, path, progress)


def clean(changes):
    """Keep what is known and actually on disk."""
    out = {}
    for name, value in (changes or {}).items():
        if name not in FIELDS or not isinstance(value, str):
            continue
        _, _, _, suffix = FIELDS[name]
        path = value if os.path.isabs(value) else os.path.join(ROOT, value)
        if os.path.basename(path) in REJECTED:
            continue                         # measured not to work; see REJECTED
        if path.endswith(suffix) and os.path.exists(path):
            out[name] = path
    return out


def all():
    global _values
    with _lock:
        if _values is None:
            _values = defaults()
            if os.path.exists(PATH):
                try:
                    with open(PATH, encoding='utf-8') as f:
                        _values.update(clean(json.load(f)))
                except ValueError:
                    pass                     # a broken file is not worth refusing to start over
        return dict(_values)


def get(name):
    return all()[name]


def update(changes):
    global _values
    kept = clean(changes)
    with _lock:
        _values = {**all(), **kept}
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({k: os.path.relpath(v, ROOT) for k, v in _values.items()}, f, indent=1)
        os.replace(tmp, PATH)                # never a half-written settings file
        return dict(_values)


def reset():
    return update(defaults())


def missing():
    """Which chosen files are not there, so the menu can say so instead of a server failing to start."""
    return {name: path for name, path in all().items() if not os.path.exists(path)}
