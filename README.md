# Kreyòl san entènèt

Haitian Creole speech tools that run on a Mac with no internet connection: speech to text with the
fine-tuned Whisper model **oswald-large-v3-turbo-m3** (whisper.cpp), and text to speech with
**qwen3-tts-1.7b-kreyol** (llama.cpp). Both run on the Mac's GPU through Metal.

| Command | What it does |
|---|---|
| `./start-page.sh` | Opens a page with both tools: **Koute** (record or upload speech, get text with timestamps) and **Pale** (type text, hear it in one of five voices or your own). Ctrl+C in the terminal stops it. |
| `./transcribe.sh recording.m4a` | Speech to text on the command line. Any format ffmpeg reads. |
| `./speak.sh "Bonjou!" kreyol_m1 out.wav` | Text to speech on the command line. Voices: `kreyol_f1`, `kreyol_f2`, `kreyol_f3`, `kreyol_m1`, `kreyol_v5`. |

Everything binds to 127.0.0.1: the page, the model servers behind it, and the audio never leave the machine.

## Setup

Tested on an M3 Pro (18 GB) with macOS 14. You need the Xcode command line tools, and:

```bash
brew install cmake ffmpeg
```

```bash
pip3 install numpy soundfile "huggingface_hub[hf_xet]"
```

The two model repos are private, so log in with an account that can read them:

```bash
hf auth login
```

Then clone this repo and run `setup.sh`. It builds whisper.cpp and llama.cpp at the commits they were
tested with, applies `llama-tts-serve.patch`, and downloads the models (about 2 GB).

```bash
git clone https://github.com/jsbeaudry/kreyol-offline.git ~/kreyol-offline
```

```bash
~/kreyol-offline/setup.sh
```

On the M3 Pro the builds took 88 s, before the download. Running it again skips whatever is already
built or downloaded. Build output goes to `setup.log`.

## What to expect

- Speech to text: about 2 s for a 10 s recording, once the page shows **Koute · pare**.
- Text to speech: a short sentence in about 2–3 s, a long one in about 5 s; faster than real time.
- The first run of each program after a build is much slower (about 25 s) while Metal compiles its GPU
  kernels. After that they are cached: `transcribe.sh` went from 26.6 s to 1.4 s on the same clip.
- Text comes back lowercase without punctuation, the form the model was trained on.
- The voice reads numbers and times as words (`8:30` → *uit è trant*); the page shows what it read.

## Accuracy

Measured on a GPU before the files were published (details in [`convert/`](convert/README.md)):

- **Speech to text**, q5_0 (574 MB): WER 4.22 / CER 1.43 on 150 held-out Bible clips, against 4.37 / 1.47
  for the original PyTorch model. Over all 304 test clips (Bible, radio, CMU), its transcripts differ from
  PyTorch's by 0.27% of characters.
- **Text to speech**, Q4_K_M talker + Q8_0 mmproj (1.5 GB): CER 2.80 when 72 generated utterances are
  transcribed back, against 3.28 for the PyTorch model. No measurable loss.

## How the page works

`app/server.py` is a Python standard-library server (plus numpy and soundfile) in front of two programs
that stay loaded:

- **Koute:** whisper.cpp's `whisper-server` on the page's port + 1. The model was fine-tuned without
  timestamp tokens, so its own timestamps are unreliable; instead, Silero VAD finds the speech, pauses
  of up to 2 s stay inside a line (at most 28 s long), and each line is transcribed on its own.
- **Pale:** one `llama-tts` reading jobs from stdin (see the patch below). `kreyol-tts/kreyol_text.py`
  spells out numbers and splits long text into sentences first. **Vwa pa w** clones a voice from a
  recording of at least 3 s (it uses the first 15 s).

`PORT=8190 ./start-page.sh` runs it on another port.

## The voice-model patch

`llama-tts-serve.patch` adds `-p -` to llama.cpp's `llama-tts`: it loads the model once and reads one job
per line from stdin. The page uses it to keep the voice model loaded, which saves about 1.1 s on every
reading (a short sentence: 2.0 s instead of 3.1 s). Two jobs with the same text, voice and seed give
bit-identical audio, so nothing leaks from one reading to the next.

`setup.sh` applies it. After updating llama.cpp yourself, apply it again and rebuild:

```bash
git -C llama.cpp apply ../llama-tts-serve.patch
```

```bash
cmake --build llama.cpp/build -j --target llama-tts
```

If it no longer applies, the page still works: it notices that `llama-tts` has no job loop and starts
one process per reading, about 1 s slower. The **Pale** lamp's tooltip says which mode is active.

## What is in here

- `app/`: the page (`index.html`) and its server (`server.py`).
- `start-page.sh`, `transcribe.sh`, `speak.sh`, `setup.sh`, `llama-tts-serve.patch`.
- `convert/`: the RunPod pipeline that converted both models, checked them against PyTorch, and
  published them.

Made by `setup.sh`, not in git:

- `whisper.cpp/`, `llama.cpp/`: built from source with Metal (GPU) support.
- `models/`: `ggml-oswald-m3-q5_0.bin` (speech to text) and the Silero voice-activity detector, from
  `jsbeaudry/oswald-large-v3-turbo-m3-ggml`.
- `kreyol-tts/`: the Kreyòl talker and mmproj GGUFs, five reference voices and the text normaliser, from
  `jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF`.
