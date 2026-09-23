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

# What is published, so the menu can offer a model that is not on this machine yet and fetch it. Sizes
# are megabytes as the Hub reports them, shown before anything is downloaded because the difference
# between these is gigabytes.
REPOS = {'stt_model': 'jsbeaudry/oswald-large-v3-turbo-m3-ggml',
         'tts_model': 'jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF',
         'tts_mmproj': 'jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF'}
CATALOGUE = {
    'stt_model': [('ggml-oswald-m3-q5_0.bin', 574, 'recommended'),
                  ('ggml-oswald-m3-q8_0.bin', 874, ''),
                  ('ggml-oswald-m3-f16.bin', 1625, 'reference')],
    'tts_model': [('qwen3-tts-1.7b-kreyol-Q4_K_M.gguf', 1036, 'recommended'),
                  ('qwen3-tts-1.7b-kreyol-Q3_K_M.gguf', 826, 'smallest that works'),
                  ('qwen3-tts-1.7b-kreyol-Q8_0.gguf', 1848, ''),
                  ('qwen3-tts-1.7b-kreyol-f16.gguf', 3473, 'reference; re-quantise from this')],
    'tts_mmproj': [('mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf', 493, 'recommended'),
                   ('mmproj-qwen3-tts-1.7b-kreyol-f16.gguf', 701, '')],
}

# Quantisations measured not to work, listed so that one sitting in the folder cannot be chosen by
# mistake. Q2_K was made and tested on 2026-09-23: asked for a seven-second sentence it produced 41
# seconds of "li li li li" with Chinese and Cyrillic in it. At 1.7B parameters two bits take away the
# model's grounding and its ability to stop. See convert/README.md.
REJECTED = {'qwen3-tts-1.7b-kreyol-Q2_K.gguf': 'unusable: loops and never stops',
            'qwen3-tts-1.7b-kreyol-IQ2_XXS.gguf': 'unusable: two bits is too few for this model',
            'qwen3-tts-1.7b-kreyol-IQ2_S.gguf': 'unusable: two bits is too few for this model'}

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


def options(name):
    """Everything this setting could be: published files, and anything else already in the folder.

    A published file that is not on this machine is still offered, with its size, so choosing between
    quantisations does not mean first knowing they exist. Downloading one is `fetch`.
    """
    _, _, folder, _ = FIELDS[name]
    out, seen = [], set()
    for filename, mb, note in CATALOGUE.get(name, []):
        path = os.path.join(folder, filename)
        seen.add(path)
        out.append({'path': path, 'label': label(path), 'mb': mb, 'note': note,
                    'local': os.path.exists(path), 'rejected': REJECTED.get(filename)})
    for path in choices(name):                  # anything dropped in by hand, or an older download
        if path not in seen:
            size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
            out.append({'path': path, 'label': label(path), 'mb': round(size), 'note': '',
                        'local': True, 'rejected': REJECTED.get(os.path.basename(path))})
    return out


def room_for(mb, folder):
    """Whether there is disk for a download, with a little to spare."""
    import shutil
    try:
        return shutil.disk_usage(folder).free > mb * 1e6 * 1.15
    except Exception:
        return True                             # cannot tell; let the download itself complain


def fetch(name, path, progress=None):
    """Download a published file into the folder this setting reads from.

    Returns the path. `progress` is called with a fraction, worked out by watching the part-file
    huggingface_hub writes, because it offers no callback of its own.
    """
    from huggingface_hub import hf_hub_download

    _, _, folder, _ = FIELDS[name]
    filename = os.path.basename(path)
    wanted = next((mb for f, mb, _ in CATALOGUE.get(name, []) if f == filename), 0)
    if wanted and not room_for(wanted, folder):
        raise RuntimeError(f'not enough disk for {filename} ({wanted:,} MB)')

    stop = threading.Event()

    def watch():
        while not stop.wait(0.5):
            biggest = 0
            for root, _, files in os.walk(os.path.join(folder, '.cache')):
                for f in files:
                    if f.endswith('.incomplete'):
                        try:
                            biggest = max(biggest, os.path.getsize(os.path.join(root, f)))
                        except OSError:
                            pass
            if progress and wanted:
                progress(min(0.99, biggest / (wanted * 1e6)))

    if progress:
        threading.Thread(target=watch, daemon=True).start()
    try:
        got = hf_hub_download(REPOS[name], filename, local_dir=folder)
    finally:
        stop.set()
    if progress:
        progress(1.0)
    return got


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
