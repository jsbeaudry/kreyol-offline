"""PyTorch baseline: the clone model reads every sentence in each eval voice, as the Space does (venv-tts)."""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVAL_VOICES, OUT_TTS, SENTENCES, SENTENCES_EXTRA, W   # noqa: E402
from kreyol_text import normalize                                        # noqa: E402

BIG = bool(os.environ.get("BIG"))   # all 36 sentences with fresh seeds, for a comparison that clears the noise
SENTS = SENTENCES + SENTENCES_EXTRA if BIG else SENTENCES
SEED = int(os.environ.get("SEED_OFFSET", "500" if BIG else "0"))   # a second seed set measures run-to-run noise
out = f"{W}/tts_eval/pytorch" + ("-big" if BIG else f"-s{SEED}" if SEED else "")
os.makedirs(out, exist_ok=True)
# The same checkpoint convert_tts.sh converted, loaded from its local copy.
model = Qwen3TTSModel.from_pretrained(f"{W}/tts", device_map="cuda", dtype=torch.bfloat16)
timing = {}
for voice in EVAL_VOICES:
    ref, sr = sf.read(f"{OUT_TTS}/voices/{voice}.wav", dtype="float32")
    for i, s in enumerate(SENTS):
        torch.manual_seed(SEED + i)
        started = time.time()
        wavs, osr = model.generate_voice_clone(text=[normalize(s)], ref_audio=(ref, sr), x_vector_only_mode=True,
                                               max_new_tokens=400)
        secs = time.time() - started
        w = wavs[0].float().cpu().numpy() if torch.is_tensor(wavs[0]) else np.asarray(wavs[0], dtype=np.float32)
        sf.write(f"{out}/{voice}_{i:02d}.wav", w, osr)
        timing[f"{voice}_{i:02d}"] = {"seconds": round(secs, 2), "audio": round(len(w) / osr, 2)}
    print(f"{voice}: {len(SENTS)} sentences", flush=True)
json.dump(timing, open(f"{out}/timing.json", "w"), indent=2)
