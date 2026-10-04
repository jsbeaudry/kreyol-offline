"""Qwen-ASR CER for one or more folders of NN.wav made from val/onnx_int8/lines.json."""
import io, json, os, sys
import soundfile as sf
sys.path.insert(0, os.path.expanduser("~/kreyol-offline/klara"))
sys.path.insert(0, os.path.expanduser("~/Documents/Youtube-Ht-Data/scripts/whisper_ft"))
import qwen_asr
from normalize import metric_form

M = os.path.expanduser("~/kreyol-offline/kreyol-asr/qwen3-asr-0.6b-kreyol-v2-Q4_K_M.gguf")
lines = json.load(open("val/onnx_int8/lines.json"))


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


qwen_asr.ensure(M)
for folder in sys.argv[1:]:
    err = tot = 0
    for i, text in enumerate(lines):
        x, sr = sf.read(f"{folder}/{i:02d}.wav")
        buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
        ref = metric_form(text)
        e = lev(ref, metric_form(qwen_asr.transcribe(buf.getvalue(), M) or ""))
        err += min(e, len(ref)); tot += len(ref)
    print(f"{folder:22} CER {100 * err / tot:.1f}%")
qwen_asr.stop()
