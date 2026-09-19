"""llama.cpp runs: each talker quantisation reads every sentence in each eval voice with llama-tts.

Sampling matches Qwen3-TTS's own defaults (temperature 0.9, top-k 50, top-p 1.0, repetition penalty 1.05)
so the comparison with the PyTorch baseline is about the conversion, not the decoding settings. -c 2048:
without it llama-tts allocates a 32k-token KV cache (3.5 GB) for a few hundred audio tokens.
"""
import json
import os
import subprocess
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVAL_VOICES, OUT_TTS, SENTENCES, SENTENCES_EXTRA, TTS_NAME, W   # noqa: E402
from kreyol_text import normalize                                                  # noqa: E402

TTS = f"{W}/llama.cpp/build/bin/llama-tts"
MMPROJ_Q = os.environ.get("MMPROJ", "Q8_0")   # the audio decoder's own precision, tested separately
MMPROJ = f"{OUT_TTS}/mmproj-{TTS_NAME}-{MMPROJ_Q}.gguf"
# llama-tts defaults Qwen3-TTS to English; "auto" (patched llama.cpp) is the prompt the model was tuned with
LANG = os.environ.get("TTS_LANG", "")
BIG = bool(os.environ.get("BIG"))   # all 36 sentences with fresh seeds, for a comparison that clears the noise
SENTS = SENTENCES + SENTENCES_EXTRA if BIG else SENTENCES
SEED = 500 if BIG else 0

for quant in sys.argv[1:] or ["Q8_0", "Q4_K_M"]:
    out = f"{W}/tts_eval/{quant}" + ("" if MMPROJ_Q == "Q8_0" else f"-mm{MMPROJ_Q}") + (f"-{LANG}" if LANG else "") + ("-big" if BIG else "")
    os.makedirs(out, exist_ok=True)
    timing = {}
    for voice in EVAL_VOICES:
        for i, s in enumerate(SENTS):
            wav = f"{out}/{voice}_{i:02d}.wav"
            cmd = [TTS, "-m", f"{OUT_TTS}/{TTS_NAME}-{quant}.gguf", "--mmproj", MMPROJ, "-p", normalize(s),
                   "--tts-speaker-file", f"{OUT_TTS}/voices/{voice}.wav", "-o", wav, "-c", "2048", "-ngl", "99",
                   "--temp", "0.9", "--top-k", "50", "--top-p", "1.0", "--repeat-penalty", "1.05", "--seed", str(SEED + i)]
            cmd += ["--tts-lang", LANG] if LANG else []
            started = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            secs = time.time() - started
            if r.returncode != 0 or not os.path.exists(wav):
                print(f"FAILED {quant} {voice} {i}: exit {r.returncode}\n{r.stderr[-1500:]}", flush=True)
                sys.exit(1)
            info = sf.info(wav)
            timing[f"{voice}_{i:02d}"] = {"seconds": round(secs, 2), "audio": round(info.frames / info.samplerate, 2)}
        print(f"{quant} {voice}: {len(SENTS)} sentences", flush=True)
    json.dump(timing, open(f"{out}/timing.json", "w"), indent=2)
