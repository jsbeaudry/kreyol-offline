# Kreyòl speech models for offline use: whisper.cpp and llama.cpp

These scripts ran on a RunPod GPU pod. They converted both speech models to run without a GPU server or a
network, checked every file against the original PyTorch model, and pushed the results as private repos:

| Model | Converted to | Repo |
|---|---|---|
| ASR `oswald-large-v3-turbo-m3` | whisper.cpp ggml (f16, q8_0, q5_0) + Silero VAD | [oswald-large-v3-turbo-m3-ggml](https://huggingface.co/jsbeaudry/oswald-large-v3-turbo-m3-ggml) |
| TTS `qwen3-tts-1.7b-kreyol` | llama.cpp GGUF talker (f16, Q8_0, Q4_K_M) + mmproj (f16, Q8_0) | [qwen3-tts-1.7b-kreyol-GGUF](https://huggingface.co/jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF) |

| File | Purpose |
|---|---|
| `setup.sh` | Builds whisper.cpp and llama.cpp with CUDA for the pod's GPU, and three venvs that must not share packages |
| `podenv.sh` | Loads the pod's `HF_TOKEN` secret into an SSH session without printing it |
| `convert_asr.sh` | m3 → ggml at three sizes, after proving its tokenizer matches base turbo's |
| `convert_tts.sh` | The clone-capable TTS model → talker and mmproj GGUFs |
| `eval_asr.py` | 304 held-out clips through PyTorch and each ggml size: error rates and difference from PyTorch |
| `make_voices.py`, `pick_voices.py` | A reference clip per named voice, the best of three takes by ASR |
| `tts_pytorch.py`, `tts_llama.py`, `eval_tts.py` | The same sentences through PyTorch and llama.cpp, scored by transcribing them |
| `asr_lib.py`, `common.py`, `normalize.py` | PyTorch m3 reference, shared sentences and paths, the ASR text normaliser |
| `say.py`, `kreyol_text.py` | Shipped with the TTS files: spells out numbers, splits long text, runs `llama-tts`, joins the audio |
| `push_gguf.py`, `card_asr.md`, `card_tts.md` | Both repos, one commit each, with the measured cards |
| `llama-tts-lang-auto.patch` | Optional: adds `--tts-lang auto` to llama.cpp. Measured to make no difference for this model |

## Run on a pod

RTX 4090, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 100 GB container disk, the `HF_TOKEN` secret.
Copy this folder to `/workspace/gguf` and run everything from there; outputs go to `/root/work`.

```bash
./setup.sh                      # ~25 min, most of it llama.cpp's CUDA kernels
./convert_asr.sh & ./convert_tts.sh & wait
python eval_asr.py
python make_voices.py && python pick_voices.py          # venv-tts, then venv-whisper
python tts_pytorch.py && python tts_llama.py && python eval_tts.py
BIG=1 ...                       # 36 sentences instead of 12; see "Noise" below
python push_gguf.py
```

The run on 2026-09-19 took 1 h 19 min of pod time, about $0.98.

## How small can the talker go?

Asked on 2026-09-23, answered on an M3 Pro rather than a pod: `llama-quantize` from the f16, then
`compare_quants.py` — the same sentences, voices and metric `eval_tts.py` uses, but judged with the
ggml m3 through `whisper-server` because that is what is on a laptop. 72 utterances per variant, as
"Noise" below requires.

| Variant | Size | CER | WER | CER f1 | CER m1 | Verdict |
|---|---|---|---|---|---|---|
| Q4_K_M | 1,036 MB | **2.95** | 10.97 | 3.46 | 2.44 | the shipped default |
| Q3_K_M | 826 MB | 3.51 | 12.90 | 4.43 | 2.59 | works, 20% smaller, slightly worse |
| Q2_K | 632 MB | — | — | — | — | **unusable** |

**Two bits destroys this model.** Asked for a seven-second sentence, Q2_K produced 41 seconds of audio
that transcribes as `bot li li li li li li ... gen怎么 li li ... дух solèy telesè`: degenerate looping,
Chinese and Cyrillic, and no ability to stop. It was not worth the 72-utterance run; one sentence
settled it. 1.7B parameters is too few to survive two bits. IQ2 with an importance matrix built from
Kreyòl calibration text might do better, but that is a project, not a quantisation.

**Three bits works but is not obviously worth it.** Q3_K_M saves 210 MB and costs +0.56 CER. Note what
"Noise" says: at 24 utterances PyTorch alone swung 1.0 CER between seed sets, so at 72 the sampling
noise is roughly 0.6 — the same size as the difference being measured. Read this as "Q3_K_M is not
clearly worse and not clearly equal", not as a measured 19% degradation. Deciding properly would take
several hundred utterances, for 210 MB.

`dikte/settings.py` carries Q2_K in a `REJECTED` table so a copy left in the folder cannot be chosen
from the menu by mistake.

## Results

**ASR**, 304 held-out clips, greedy decoding:

| Variant | Size | Bible WER / CER | Differs from PyTorch | Speed (4090) |
|---|---|---|---|---|
| PyTorch fp16 | 1,617 MB | 4.37 / 1.47 | — | 118× real time |
| ggml f16 | 1,625 MB | 4.46 / 1.47 | 0.05% CER | 121× |
| ggml q8_0 | 874 MB | 4.40 / 1.47 | 0.11% | 135× |
| **ggml q5_0** | **574 MB** | **4.22 / 1.43** | 0.27% | 140× |

**TTS**, 36 sentences × 2 voices, CER through m3:

| Variant | CER |
|---|---|
| PyTorch | 3.28 |
| **Q4_K_M talker + Q8_0 mmproj**, stock `llama-tts` | **2.80** |
| same, `--tts-lang fr` | 2.92 |
| same, patched `--tts-lang auto` | 2.77 |

**On the M3 Pro** (Metal): ASR 1.2–1.5 s for a 7–12 s clip; TTS generates at 1.36× real time. The first run
after a build takes much longer while Metal compiles its kernels.

## Gotchas

- **m3 has no `vocab.json`.** transformers 5 saves only `tokenizer.json`, and whisper.cpp's converter needs
  `vocab.json` and `added_tokens.json`. The tokenizer is untouched by fine-tuning, so `convert_asr.sh` takes
  them from `openai/whisper-large-v3-turbo` after asserting the vocabularies match (50,257 + 1,609 tokens).
- **whisper-cli defaults to English.** Always pass `-l ht`.
- **Use `--vad` with m3.** Without it the converted model dropped a whole sentence after 6 s of silence, and
  wrote *"anpil moun ki te genyen moun ki te genyen…"* over 8 s of silence. With it, both were right.
- **llama.cpp cannot select a named speaker.** It only clones from `--tts-speaker-file`, so the five voices of
  the `-voices` model ship as reference clips for the clone model. Issue #29088 (CustomVoice conversion fails
  on a missing `speaker_encoder_config`) would not affect the `-voices` model, which has that key.
- **`llama-tts --tts-lang` defaults to English** and always inserts a language token, while this model was
  tuned with none. It looked like the cause of a quality gap, but at 72 utterances the flag made no difference.
- **Sampling noise is large.** With 24 utterances per variant, PyTorch alone moved from 1.5% to 2.5% CER
  between seed sets. Compare TTS variants on 72+ utterances before believing a difference.
- **`llama-tts -c 2048`.** Without it, a 32k-token KV cache (3.5 GB) is allocated for a few hundred tokens.
- **`hf download --include` takes one pattern per flag.** `--include "a" "b"` turns `b` into a literal filename.
- **Pod image:** `nvcc` is in `/usr/local/cuda/bin` but not on `PATH`, and the CPU quota is in cgroup v1
  (`cpu.cfs_quota_us`), not `nproc`.
