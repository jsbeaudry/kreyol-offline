"""Score every TTS run by transcribing it with the PyTorch m3: CER against the text it was asked to read.

This is how the TTS models were judged before (CER through an ASR model), so the numbers are comparable.
The same sentences, voices and ASR are used for every variant, so differences come from the conversion;
number words can be spelled differently by the text normaliser and the ASR, which costs every variant alike.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_lib import rates, transcribe                  # noqa: E402
from common import EVAL_VOICES, OUT_TTS, SENTENCES, SENTENCES_EXTRA, W  # noqa: E402
from kreyol_text import normalize                      # noqa: E402

BIG = bool(os.environ.get("BIG"))   # score the 36-sentence runs
SENTS = SENTENCES + SENTENCES_EXTRA if BIG else SENTENCES
report = {}
for variant in sys.argv[1:] or ["pytorch", "Q8_0", "Q4_K_M"]:
    d = f"{W}/tts_eval/{variant}"
    timing = json.load(open(f"{d}/timing.json"))
    keys = [f"{v}_{i:02d}" for v in EVAL_VOICES for i in range(len(SENTS))]
    hyps = dict(zip(keys, transcribe([f"{d}/{k}.wav" for k in keys])))
    row = {}
    for voice in EVAL_VOICES + ["all"]:
        ks = [k for k in keys if voice == "all" or k.startswith(voice + "_")]
        refs = [normalize(SENTS[int(k[-2:])]) for k in ks]
        row[voice] = rates(refs, [hyps[k] for k in ks])
    gen = sum(timing[k]["seconds"] for k in keys)
    audio = sum(timing[k]["audio"] for k in keys)
    row["rtf"] = round(gen / audio, 2)
    row["audio_seconds"] = round(audio, 1)
    row["worst"] = sorted(((rates([normalize(SENTS[int(k[-2:])])], [hyps[k]])["cer"], k, hyps[k]) for k in keys),
                          reverse=True)[:3]
    report[variant] = row
json.dump(report, open(f"{OUT_TTS}/eval{'-big' if BIG else ''}.json", "w"), indent=2, ensure_ascii=False)

print("| variant | CER all | " + " | ".join(f"CER {v}" for v in EVAL_VOICES) + " | real-time factor | audio |")
print("|---|---|" + "---|" * len(EVAL_VOICES) + "---|---|")
for variant, r in report.items():
    print(f"| {variant} | {r['all']['cer']} | " + " | ".join(str(r[v]["cer"]) for v in EVAL_VOICES) +
          f" | {r['rtf']} | {r['audio_seconds']} s |")
for variant, r in report.items():
    print(f"\nworst {variant}: " + "; ".join(f"{k} CER {c}: {h[:70]!r}" for c, k, h in r["worst"]))
