"""Pocket v1 as INT8 ONNX vs the PyTorch original: same 20 held-out sentences, same voice, Qwen-ASR CER.

    .venv/bin/python validate.py BUNDLE_DIR OUT_DIR
"""
import json, os, sys, time
import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import pocket_tts_onnx  # noqa: E402

bundle, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
TOKENIZER = os.path.expanduser("~/kreyol-offline/models/pocket-kreyol/tokenizer.json")
VOICE = os.environ.get("VOICE") or os.path.expanduser("~/kreyol-offline/models/pocket-kreyol/voices/reel.safetensors")


class OurTokenizer:
    """Our fine-tune's Unigram tokenizer behind the two sentencepiece calls the runtime makes."""
    def __init__(self, path):
        self.tk = Tokenizer.from_file(path)

    def Encode(self, text):
        return self.tk.encode(text).ids

    def Decode(self, ids):
        return self.tk.decode(list(ids))


lines = json.load(open(os.path.expanduser("~/kreyol-offline/runpod/pocket_tts/cmu_lines.json")))
lines = [l if isinstance(l, str) else l.get("text", "") for l in lines][:20]
tts = pocket_tts_onnx.PocketTTSOnnx(models_dir=os.path.dirname(bundle), language=os.path.basename(bundle), precision=os.environ.get("PRECISION", "int8"))
tts.tokenizer = OurTokenizer(TOKENIZER)
voice = np.load(VOICE) if VOICE.endswith('.npy') else VOICE
times = []
for i, text in enumerate(lines):
    t = time.time()
    audio = tts.generate(text, voice)
    times.append((time.time() - t, len(audio) / 24000))
    sf.write(f"{out}/{i:02d}.wav", np.asarray(audio, dtype=np.float32), 24000)
    print(f"{i:02d} {len(audio) / 24000:.2f}s audio in {times[-1][0]:.2f}s", flush=True)
json.dump(lines, open(f"{out}/lines.json", "w"), ensure_ascii=False)
print(f"RTF {sum(a for a, _ in times) / sum(b for _, b in times):.2f}")
