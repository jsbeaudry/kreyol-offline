"""Keep the take of each voice that m3 transcribes closest to REF_TEXT, 4-14 s long (venv-whisper)."""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_lib import duration, rates, transcribe    # noqa: E402
from common import OUT_TTS, REF_TEXT, VOICES, W    # noqa: E402
from kreyol_text import normalize                  # noqa: E402

cands = sorted(f"{W}/voices_cand/{f}" for f in os.listdir(f"{W}/voices_cand") if f.endswith(".wav"))
hyps = dict(zip(cands, transcribe(cands)))
os.makedirs(f"{OUT_TTS}/voices", exist_ok=True)
chosen = {}
for spk in VOICES:
    takes = []
    for path in [c for c in cands if os.path.basename(c).startswith(spk + "_")]:
        secs = duration(path)
        cer = rates([normalize(REF_TEXT)], [hyps[path]])["cer"]
        takes.append((cer if 4 <= secs <= 14 else 999, secs, path))
    cer, secs, path = min(takes)
    shutil.copy(path, f"{OUT_TTS}/voices/{spk}.wav")
    chosen[spk] = {"cer": cer, "seconds": round(secs, 1), "take": os.path.basename(path), "asr": hyps[path]}
    print(f"{spk}: take {os.path.basename(path)}, {secs:.1f} s, CER {cer}  <- {hyps[path]}", flush=True)
json.dump({"text": REF_TEXT, "voices": chosen}, open(f"{OUT_TTS}/voices/voices.json", "w"), indent=2, ensure_ascii=False)
