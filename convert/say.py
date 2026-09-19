"""Speak Kreyòl text offline with llama.cpp's llama-tts.

llama-tts reads one prompt and stops after about 40 s of audio, and it reads digits unpredictably. This
spells out numbers and times the way the model was trained (kreyol_text.normalize), splits long text at
sentence boundaries, synthesises each piece in the chosen voice, and joins them with a short pause.

Usage: python say.py "Bonjou, kijan ou ye?" -o bonjou.wav [--voice voices/kreyol_m1.wav] [--llama-tts PATH]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

from kreyol_text import normalize, split_text

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", help="Kreyòl text, or - to read it from stdin")
    p.add_argument("-o", "--out", default="out.wav")
    p.add_argument("--voice", default=os.path.join(HERE, "voices", "kreyol_f1.wav"),
                   help="any clean 5-15 s Kreyòl clip works; voices/ holds the five fine-tuned voices")
    p.add_argument("--model", default=os.path.join(HERE, "qwen3-tts-1.7b-kreyol-Q4_K_M.gguf"))
    p.add_argument("--mmproj", default=os.path.join(HERE, "mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf"))
    p.add_argument("--llama-tts", default=shutil.which("llama-tts") or "llama-tts")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    text = sys.stdin.read() if args.text == "-" else args.text
    chunks = split_text(normalize(text))
    if not chunks:
        sys.exit("no text to speak")
    pieces, sr = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for i, chunk in enumerate(chunks):
            wav = os.path.join(tmp, f"{i:03d}.wav")
            # -c 2048: llama-tts otherwise allocates a 32k-token KV cache (3.5 GB) for a few hundred tokens.
            # Sampling is Qwen3-TTS's own default, the setting these files were measured with.
            cmd = [args.llama_tts, "-m", args.model, "--mmproj", args.mmproj, "-p", chunk,
                   "--tts-speaker-file", args.voice, "-o", wav, "-c", "2048", "-ngl", "99",
                   "--temp", "0.9", "--top-k", "50", "--top-p", "1.0", "--repeat-penalty", "1.05",
                   "--seed", str(args.seed + i)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(wav):
                sys.exit(f"llama-tts failed on chunk {i + 1}:\n{r.stderr[-1200:]}")
            audio, sr = sf.read(wav, dtype="float32")
            pieces += [audio, np.zeros(int(0.25 * sr), dtype=np.float32)]
            print(f"  {i + 1}/{len(chunks)}: {len(audio) / sr:.1f} s  {chunk[:60]}", flush=True)
    sf.write(args.out, np.concatenate(pieces[:-1]), sr)
    print(f"{args.out}: {sum(len(x) for x in pieces[:-1]) / sr:.1f} s")


if __name__ == "__main__":
    main()
