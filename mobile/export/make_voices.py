"""Voice embeddings for the app, computed exactly as pocket-tts 3.3.0 conditions on an audio prompt:
Mimi latent of the clip -> speaker_proj. Saved WITHOUT the BOS vector (the runtime prepends it)."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio

cfg = os.path.expanduser("~/kreyol-offline/mobile/export/cfg/kreyol_v1_full.yaml")
model = TTSModel.load_model(config=cfg).eval()
clips = os.path.expanduser("~/kreyol-offline/samples/pocket_kreyol/voices")
index = {}
with torch.no_grad():
    for name in ("reel", "vocal", "female1", "female2", "man1", "man2"):
        audio, sr = audio_read(Path(f"{clips}/voice_{name}.wav"))
        audio = convert_audio(audio, sr, model.config.mimi.sample_rate, 1)
        emb = model._encode_audio(audio.unsqueeze(0)).float().numpy()      # (1, T, 1024)
        emb.astype("<f4").tofile(f"voices/{name}.f32")
        np.save(f"voices/{name}.npy", emb)
        index[name] = {"file": f"{name}.f32", "shape": list(emb.shape)}
        print(name, emb.shape)
json.dump(index, open("voices/voices.json", "w"), indent=1)
