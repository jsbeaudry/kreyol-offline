"""The models this kit can run, where they live, and which ones are published: one table.

Both the page and Dikte let you choose a model, so both need to know what exists, what may be fetched,
and what has been measured not to work. Keeping two copies of that would eventually let one of them
offer a model the other knows is broken, so this module is the single copy and both read it.

`REJECTED` is the part that matters most. A quantisation that does not work does not raise an error —
it produces speech that loops and never stops — so the finding is recorded here where the menus read
it, not only in convert/README.md where someone would have to go looking.
"""
import glob
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "stt_model": {
        "title": "Speech to text", "kreyol": "Modèl tèks",
        "folder": "models", "suffix": ".bin",
        "default": "models/ggml-oswald-m3-q5_0.bin",
        "repo": "jsbeaudry/oswald-large-v3-turbo-m3-ggml",
        "published": [("ggml-oswald-m3-q5_0.bin", 574, "recommended"),
                      ("ggml-oswald-m3-q8_0.bin", 874, ""),
                      ("ggml-oswald-m3-f16.bin", 1625, "reference")],
    },
    "tts_model": {
        "title": "Voice", "kreyol": "Vwa a",
        "folder": "kreyol-tts", "suffix": ".gguf",
        "default": "kreyol-tts/qwen3-tts-1.7b-kreyol-Q4_K_M.gguf",
        "repo": "jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF",
        "published": [("qwen3-tts-1.7b-kreyol-Q4_K_M.gguf", 1036, "recommended"),
                      ("qwen3-tts-1.7b-kreyol-Q3_K_M.gguf", 826, "smallest that works"),
                      ("qwen3-tts-1.7b-kreyol-Q8_0.gguf", 1848, ""),
                      ("qwen3-tts-1.7b-kreyol-f16.gguf", 3473, "reference; re-quantise from this")],
    },
    "tts_mmproj": {
        "title": "Voice projector", "kreyol": "Dekodè vwa a",
        "folder": "kreyol-tts", "suffix": ".gguf",
        "default": "kreyol-tts/mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf",
        "repo": "jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF",
        "published": [("mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf", 493, "recommended"),
                      ("mmproj-qwen3-tts-1.7b-kreyol-f16.gguf", 701, "")],
    },
}
# Which server has to be loaded again when one of these changes.
RELOADS = {"stt_model": "asr", "tts_model": "tts", "tts_mmproj": "tts"}

# Measured not to work, so a copy in the folder cannot be chosen by mistake. Q2_K was made and tested
# on 2026-09-23: asked for a seven-second sentence it produced 41 seconds of "li li li li" with Chinese
# and Cyrillic in it. At 1.7B parameters two bits takes away the model's grounding and its ability to
# stop. See convert/README.md.
REJECTED = {"qwen3-tts-1.7b-kreyol-Q2_K.gguf": "unusable: loops and never stops",
            "qwen3-tts-1.7b-kreyol-IQ2_XXS.gguf": "unusable: two bits is too few for this model",
            "qwen3-tts-1.7b-kreyol-IQ2_S.gguf": "unusable: two bits is too few for this model"}


def default(name):
    return os.path.join(ROOT, MODELS[name]["default"])


def folder(name):
    return os.path.join(ROOT, MODELS[name]["folder"])


def label(path):
    """What to call a model on screen: its filename without the extension."""
    return os.path.splitext(os.path.basename(path))[0]


def usable(name, path):
    """Whether this file could actually be loaded for this setting."""
    return (os.path.basename(path) not in REJECTED
            and path.endswith(MODELS[name]["suffix"]) and os.path.exists(path))


def on_disk(name):
    """Files of the right kind in this setting's folder, with the ones that are not models removed."""
    spec = MODELS[name]
    found = sorted(glob.glob(os.path.join(folder(name), "*" + spec["suffix"])))
    if name == "tts_model":                    # the projector is not a talker, and the reverse
        found = [p for p in found if not os.path.basename(p).startswith("mmproj")]
    elif name == "tts_mmproj":
        found = [p for p in found if os.path.basename(p).startswith("mmproj")]
    elif name == "stt_model":
        # The voice-activity detector lives here too, and whisper.cpp's test fixtures are not models.
        found = [p for p in found if "silero" not in os.path.basename(p).lower()
                 and not os.path.basename(p).startswith("for-tests")]
    return found


def options(name, current=None):
    """Everything this setting could be: what is published, plus anything else in the folder.

    A published file that is not on this machine is still listed, with its size, so choosing between
    quantisations does not mean first knowing they exist.
    """
    out, seen = [], set()
    for filename, mb, note in MODELS[name]["published"]:
        path = os.path.join(folder(name), filename)
        seen.add(path)
        out.append({"path": path, "label": label(path), "mb": mb, "note": note,
                    "local": os.path.exists(path), "rejected": REJECTED.get(filename)})
    for path in on_disk(name) + ([current] if current and os.path.exists(current) else []):
        if path not in seen:
            seen.add(path)
            size = os.path.getsize(path) / 1e6
            out.append({"path": path, "label": label(path), "mb": round(size), "note": "",
                        "local": True, "rejected": REJECTED.get(os.path.basename(path))})
    return out


def room_for(mb, where):
    """Whether there is disk for a download, with a little to spare."""
    import shutil
    try:
        return shutil.disk_usage(where).free > mb * 1e6 * 1.15
    except OSError:
        return True                            # cannot tell; let the download itself complain


def fetch(name, path, progress=None):
    """Download a published file into the folder this setting reads from.

    `progress` is called with a fraction, worked out by watching the part-file huggingface_hub writes,
    because it offers no callback of its own.
    """
    from huggingface_hub import hf_hub_download

    into, filename = folder(name), os.path.basename(path)
    wanted = next((mb for f, mb, _ in MODELS[name]["published"] if f == filename), 0)
    if wanted and not room_for(wanted, into):
        raise RuntimeError(f"not enough disk for {filename} ({wanted:,} MB)")

    stop = threading.Event()

    def watch():
        while not stop.wait(0.5):
            biggest = 0
            for root, _, files in os.walk(os.path.join(into, ".cache")):
                for f in files:
                    if f.endswith(".incomplete"):
                        try:
                            biggest = max(biggest, os.path.getsize(os.path.join(root, f)))
                        except OSError:
                            pass
            if progress and wanted:
                progress(min(0.99, biggest / (wanted * 1e6)))

    if progress:
        threading.Thread(target=watch, daemon=True).start()
    try:
        return hf_hub_download(MODELS[name]["repo"], filename, local_dir=into)
    finally:
        stop.set()
        if progress:
            progress(1.0)
