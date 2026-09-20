# Kreyòl san entènèt

Haitian Creole speech tools that run on a Mac with no internet connection: speech to text with the
fine-tuned Whisper model **oswald-large-v3-turbo-m3** (whisper.cpp), and text to speech with
**qwen3-tts-1.7b-kreyol** (llama.cpp). Both run on the Mac's GPU through Metal.

| Command | What it does |
|---|---|
| `./start-page.sh` | Opens the page with all the tools below. Ctrl+C in the terminal stops it. |
| `./transcribe.sh recording.m4a` | Speech to text on the command line. Any format ffmpeg reads. |
| `./speak.sh "Bonjou!" kreyol_m1 out.wav` | Text to speech on the command line. Voices: `kreyol_f1`, `kreyol_f2`, `kreyol_f3`, `kreyol_m1`, `kreyol_v5`. |

Everything binds to 127.0.0.1: the page, the model servers behind it, the API, and the audio never leave
the machine.

## The page

| Tool | For | What running it locally adds |
|---|---|---|
| **Koute · Pale** | A recording up to 10 minutes to text; up to 2,000 characters to speech, in five voices or a voice cloned from a 3–15 s clip. | Works with no connection. |
| **Transkripsyon** | Recordings of any length, or a whole folder: interviews, radio shows, sermons, meetings, videos. Correct the text line by line while the audio plays, then export **SRT/VTT subtitles**, text, JSON, or the checked lines as **training data** (a 🤗 `audiofolder` zip: 16 kHz clips plus `metadata.csv`). | Hours of audio with no upload and no quota. Files in a folder are read where they are, never copied. Private recordings become training data without leaving the computer. The page also shows the model's word error rate on the lines you checked: a measure of it on your own material. |
| **Dokiman** | A lesson, article or announcement (pasted, `.txt`, `.md` or `.docx`) read aloud into one MP3, with pauses between paragraphs. A `.csv` gives one file per row (phone menus, radio spots): columns `non,tèks,vwa`. | No length limit or per-character cost; a school, clinic or radio station can make audio without internet. |
| **Bib la** | The Kreyòl Bible to read and hear: pick a book and chapter, tap a line to hear it from there, or let the voice read the chapter. It reads in blocks of about 110 characters and prepares the next three while the current one plays, so the reading starts after about 11 s and then runs without a break. A chapter can also become one MP3, or the lines can go straight into reading practice. | An audio Bible with no connection, in a voice you choose, and any chapter as a file to carry on a phone. |
| **Li ak mwen** | Reading practice: hear a sentence, read it aloud, and see which words the model heard (green), nearly heard (yellow) or missed (red). Five lessons built from the Space's example sentences, or your own text. | Recordings of learners, children included, are never saved or sent anywhere. |
| **API** | An OpenAI-compatible API for other apps and scripts (below). | Existing OpenAI code gets Kreyòl speech by changing one URL. |
| **Reglaj** | How a recording is cut into lines (detection level, shortest silence, pause kept inside a line, longest line, padding) and the silences the voice leaves between sentences and paragraphs. | Tune it to the recording in front of you. |

Long jobs run in the background, one at a time per model; the quick tools slip in between two pieces of
a job. A job that is stopped, or cut off by closing the server, resumes where it left off. Everything a
job makes is in `travay/<job>/` (the folder button opens it in Finder). Deleting a job deletes its copy of
the audio, never the original file. The page refuses a job that would leave less than 1 GB of disk free.

The Bib tab appears only when `app/data/bible.json` is there. That file is not part of the kit: build it
from the Bible pipeline's own files with

```bash
python3 app/make_bible.py path/to/bible.json path/to/merged_data.json
```

where the first file has the chapter text one sentence per line and the second has the Kreyòl book names.

The page follows the computer's light or dark setting; the switch next to the lamps (Otomatik, Klè, Fonse)
overrides it, and the browser remembers the choice.

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

Measured on the M3 Pro:

- Speech to text: about 2 s for a 10 s recording. A 15.6-minute recording took 95 s (10 times faster than
  real time), so an hour of audio takes about 6 minutes. A transcription keeps a 16 kHz copy of the audio
  for the editor: 115 MB per hour.
- Text to speech: a short sentence in about 2–3 s; long documents are made about 1.3–1.4 times faster
  than they play. The voices read about 10 characters a second.
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

## The API

Apps and scripts that use OpenAI's audio API can use these models instead: point them at
`http://127.0.0.1:8177/v1` with any API key. The model name is not checked (`whisper-1` and `tts-1` work).

| Endpoint | Does | Options |
|---|---|---|
| `POST /v1/audio/transcriptions` | Speech to text, up to 3 h per request | `file`; `response_format`: `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `POST /v1/audio/speech` | Text to speech, up to 4,096 characters | `input`, `voice`; `response_format`: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`; `speed` 0.25–4 |
| `GET /v1/models` | The two models | |

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8177/v1", api_key="kreyol")
with open("entrevi.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
client.audio.speech.create(model="tts-1", voice="kreyol_m1", input="Mèsi anpil!").write_to_file("mesi.mp3")
```

Voices: `kreyol_f1`, `kreyol_f2`, `kreyol_f3` (women) and `kreyol_m1`, `kreyol_v5` (men), or their labels
(`Fanm 1`, ...). OpenAI's voice names map to them, so apps with a fixed list still get a Kreyòl voice:
alloy and marin → f1, nova and coral → f2, shimmer and sage → f3, echo, ash, fable and cedar → m1, onyx,
ballad and verse → v5. Every format above was tested with the official `openai` Python client (2.0).

For Open WebUI: Admin Settings → Audio, engine **OpenAI** for both speech to text and text to speech, API
base URL as above (from Docker: `http://host.docker.internal:8177/v1`), any key, voice `kreyol_f1`.

The server only answers requests addressed to `127.0.0.1`, `localhost` or `host.docker.internal` on its
port, and refuses changes sent by other websites open in the browser. To let a web app you trust call the
API from the browser, start it with `python3 app/server.py --allow-origin http://localhost:3000`.

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

## How it works

`app/server.py` (Python standard library, plus numpy and soundfile) serves the page and keeps two
programs loaded:

- **Speech to text:** whisper.cpp's `whisper-server` on the page's port + 1. The model was fine-tuned
  without timestamp tokens, so its own timestamps are unreliable; instead, Silero VAD finds the speech,
  pauses of up to 0.8 s stay inside a line (at most 20 s long), and each line is transcribed on its own.
  Those two numbers matter: on five minutes of speech, keeping 2 s pauses and lines up to 28 s put 7.5% of
  the words inside a repeated run, one line 31% (the model looping on a long line). At 0.8 s and 20 s none
  were, and the same speech came out 100 words shorter. The Reglaj tab changes them, and a transcription
  already made can be cut again with **Refè liy yo**. Subtitles split long lines into cues of about 7 s,
  timed in proportion to their length.
- **Text to speech:** one `llama-tts` reading jobs from stdin (see the patch above).
  `kreyol-tts/kreyol_text.py` spells out numbers and splits long text into sentences first.

| File | Holds |
|---|---|
| `app/server.py` | HTTP routes, the host and origin checks, file downloads with range requests |
| `app/engine.py` | The two models, audio conversion, transcription by speech region, synthesis |
| `app/jobs.py` | Background jobs in `travay/`, the editor's saves, exports, documents |
| `app/openai_api.py` | The `/v1` API |
| `app/practice.py` | Reading practice: the lessons and the word alignment |
| `app/settings.py` | The settings the Reglaj tab changes, kept in `travay/settings.json` |
| `app/bible.py`, `app/make_bible.py` | The Bib tab's text, and the script that prepares it |
| `app/asr_normalize.py` | The text form m3 was trained and scored on (a copy of the training normaliser) |
| `app/index.html`, `app/static/` | The page: one script per tool, no outside assets |

`PORT=8190 ./start-page.sh` runs it on another port.

## What is in here

- `app/`: the page and its server (above).
- `start-page.sh`, `transcribe.sh`, `speak.sh`, `setup.sh`, `llama-tts-serve.patch`.
- `convert/`: the RunPod pipeline that converted both models, checked them against PyTorch, and
  published them.

Made by `setup.sh` or by the page, not in git:

- `whisper.cpp/`, `llama.cpp/`: built from source with Metal (GPU) support.
- `models/`: `ggml-oswald-m3-q5_0.bin` (speech to text) and the Silero voice-activity detector, from
  `jsbeaudry/oswald-large-v3-turbo-m3-ggml`.
- `kreyol-tts/`: the Kreyòl talker and mmproj GGUFs, five reference voices and the text normaliser, from
  `jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF`.
- `travay/`: the page's jobs and exports.
- `app/data/bible.json`: the Bible text for the Bib tab, if you build it.
