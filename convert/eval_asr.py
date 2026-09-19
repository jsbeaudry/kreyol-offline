"""Does the ggml conversion transcribe like the PyTorch m3? Same clips, same greedy decoding, every variant.

Clips: 150 held-out Bible test clips (printed-text references), 150 radio test clips (machine references,
so agreement rather than accuracy), and the four held-out CMU clips from the Space. Each ggml variant is
scored against the references and against PyTorch m3's own output, which isolates what conversion and
quantisation changed.
"""
import io
import json
import os
import random
import subprocess
import sys
import time

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_lib import duration, rates, transcribe       # noqa: E402
from common import CMU_EXAMPLES, OUT_ASR, W           # noqa: E402

CLIPS = f"{W}/eval_asr/clips"
CLI = f"{W}/whisper.cpp/build/bin/whisper-cli"


def fetch_test(repo, n, tag):
    api, fs = HfApi(), HfFileSystem()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet") and "/test" in f]
    rows = []
    for f in files:
        with fs.open(f"datasets/{repo}/{f}", "rb") as fh:
            rows += [r for r in pq.ParquetFile(fh).read(columns=["audio", "text", "qc_pass"]).to_pylist() if r["qc_pass"]]
    random.Random(0).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:n]):
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        path = f"{CLIPS}/{tag}_{i:04d}.wav"
        sf.write(path, audio, sr, subtype="PCM_16")
        out.append({"path": path, "ref": r["text"], "domain": tag})
    return out


def run_whisper_cpp(model, clips, tag):
    """One process for every clip; whisper-cli writes <input>.txt beside each input, so each model gets
    its own directory of links to the same audio."""
    d = f"{W}/eval_asr/{tag}"
    os.makedirs(d, exist_ok=True)
    links = []
    for c in clips:
        link = f"{d}/{os.path.basename(c['path'])}"
        if not os.path.exists(link):
            os.symlink(c["path"], link)
        links.append(link)
    cmd = [CLI, "-m", model, "-l", "ht", "-nt", "-otxt", "-bs", "1", "-bo", "1", "-np"]
    for link in links:
        cmd += ["-f", link]
    started = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    secs = time.time() - started
    return [" ".join(open(link + ".txt").read().split()) for link in links], secs


def main():
    os.makedirs(CLIPS, exist_ok=True)
    clips = fetch_test("jsbeaudry/bible-kreyol-aligned", 150, "bible")
    clips += fetch_test("jsbeaudry/radio-haiti-inter-sample1", 150, "radio")
    for name, ref in CMU_EXAMPLES.items():
        path = hf_hub_download("jsbeaudry/oswald-asr-kreyol", f"examples/{name}.wav", repo_type="space")
        clips.append({"path": path, "ref": ref, "domain": "cmu"})
    audio_s = sum(duration(c["path"]) for c in clips)
    print(f"{len(clips)} clips, {audio_s / 60:.1f} min", flush=True)

    started = time.time()
    variants = {"pytorch-fp16": (transcribe([c["path"] for c in clips]), time.time() - started)}
    for q in ("f16", "q8_0", "q5_0"):
        model = f"{OUT_ASR}/ggml-oswald-m3-{q}.bin"
        variants[f"ggml-{q}"] = run_whisper_cpp(model, clips, q)
        print(f"ggml-{q}: {variants[f'ggml-{q}'][1]:.0f} s", flush=True)

    base = variants["pytorch-fp16"][0]
    report = {}
    for name, (hyps, secs) in variants.items():
        row = {"seconds": round(secs, 1), "x_realtime": round(audio_s / secs, 1),
               "size_mb": round(os.path.getsize(f"{OUT_ASR}/ggml-oswald-m3-{name[5:]}.bin") / 1e6)
               if name.startswith("ggml") else None}
        for dom in ("bible", "radio", "cmu"):
            sel = [i for i, c in enumerate(clips) if c["domain"] == dom]
            row[dom] = rates([clips[i]["ref"] for i in sel], [hyps[i] for i in sel])
        row["vs_pytorch"] = rates(base, hyps)
        report[name] = row
    json.dump(report, open(f"{OUT_ASR}/eval.json", "w"), indent=2)

    print("\n| variant | size | bible WER / CER | radio WER / CER* | CMU WER / CER | differs from PyTorch (CER) | speed |")
    print("|---|---|---|---|---|---|---|")
    for name, r in report.items():
        size = f"{r['size_mb']:,} MB" if r["size_mb"] else "1,617 MB"
        print(f"| {name} | {size} | {r['bible']['wer']} / {r['bible']['cer']} | {r['radio']['wer']} / {r['radio']['cer']} | "
              f"{r['cmu']['wer']} / {r['cmu']['cer']} | {r['vs_pytorch']['cer']} | {r['x_realtime']}x |")
    print("\n* radio references are machine transcripts: agreement, not accuracy.")


if __name__ == "__main__":
    main()
