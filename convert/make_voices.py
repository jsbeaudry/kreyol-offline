"""Reference clips for cloning: each named voice of the -voices model reads REF_TEXT three times (venv-tts).

llama.cpp can clone a voice from a clip but cannot select a named speaker, so shipping one clean clip per
voice is how the five voices survive the conversion. pick_voices.py keeps the best take of each.
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF_TEXT, VOICES, W       # noqa: E402
from kreyol_text import normalize            # noqa: E402

out = f"{W}/voices_cand"
os.makedirs(out, exist_ok=True)
model = Qwen3TTSModel.from_pretrained("jsbeaudry/qwen3-tts-1.7b-kreyol-voices", device_map="cuda", dtype=torch.bfloat16)
for spk in VOICES:
    for take in range(3):
        torch.manual_seed(take)
        wavs, sr = model.generate_custom_voice(text=[normalize(REF_TEXT)], speaker=[spk], max_new_tokens=400)
        w = wavs[0].float().cpu().numpy() if torch.is_tensor(wavs[0]) else np.asarray(wavs[0], dtype=np.float32)
        sf.write(f"{out}/{spk}_{take}.wav", w, sr)
        print(f"{spk} take {take}: {len(w) / sr:.1f} s", flush=True)
