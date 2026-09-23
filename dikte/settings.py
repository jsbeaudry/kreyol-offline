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
import glob
import json
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'dikte/settings.json')

FIELDS = {
    # name: (default, what it is, where its kind lives, the extension)
    'stt_model': (os.path.join(ROOT, 'models/ggml-oswald-m3-q5_0.bin'),
                  'Speech to text', os.path.join(ROOT, 'models'), '.bin'),
    'tts_model': (os.path.join(ROOT, 'kreyol-tts/qwen3-tts-1.7b-kreyol-Q4_K_M.gguf'),
                  'Voice', os.path.join(ROOT, 'kreyol-tts'), '.gguf'),
    'tts_mmproj': (os.path.join(ROOT, 'kreyol-tts/mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf'),
                   'Voice projector', os.path.join(ROOT, 'kreyol-tts'), '.gguf'),
    'speaker': (os.path.join(ROOT, 'kreyol-tts/voices/kreyol_f1.wav'),
                'Who reads', os.path.join(ROOT, 'kreyol-tts/voices'), '.wav'),
}
# Changing either of these means reloading a model, so the service has to be restarted; a different
# speaker is only another reference clip and takes effect on the next reading.
NEEDS_RESTART = {'stt_model', 'tts_model', 'tts_mmproj'}

_lock = threading.RLock()
_values = None


def defaults():
    return {name: spec[0] for name, spec in FIELDS.items()}


def choices(name):
    """Every file on this machine that could go in this setting, newest-looking name first."""
    _, _, folder, suffix = FIELDS[name]
    found = sorted(glob.glob(os.path.join(folder, f'*{suffix}')))
    if name == 'tts_model':                  # the projector is not a talker
        found = [p for p in found if not os.path.basename(p).startswith('mmproj')]
    elif name == 'tts_mmproj':
        found = [p for p in found if os.path.basename(p).startswith('mmproj')]
    elif name == 'stt_model':
        # The voice-activity detector lives here too, and whisper.cpp's test fixtures are not models.
        found = [p for p in found if 'silero' not in os.path.basename(p).lower()
                 and not os.path.basename(p).startswith('for-tests')]
    current = get(name)
    if current not in found and os.path.exists(current):
        found.append(current)                # something chosen by hand, outside these folders
    return found


def label(path):
    """What to call a model in a menu: its filename without the extension."""
    return os.path.splitext(os.path.basename(path))[0]


def clean(changes):
    """Keep what is known and actually on disk."""
    out = {}
    for name, value in (changes or {}).items():
        if name not in FIELDS or not isinstance(value, str):
            continue
        _, _, _, suffix = FIELDS[name]
        path = value if os.path.isabs(value) else os.path.join(ROOT, value)
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
