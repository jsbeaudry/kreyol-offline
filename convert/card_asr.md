---
base_model: jsbeaudry/oswald-large-v3-turbo-m3
language:
  - ht
license: apache-2.0
library_name: whisper.cpp
pipeline_tag: automatic-speech-recognition
tags:
  - whisper.cpp
  - ggml
  - haitian-creole
  - kreyol
  - offline
---

# oswald-large-v3-turbo-m3 for whisper.cpp

Haitian Creole (Kreyòl ayisyen) speech recognition that runs offline with [whisper.cpp](https://github.com/ggml-org/whisper.cpp): [`jsbeaudry/oswald-large-v3-turbo-m3`](https://huggingface.co/jsbeaudry/oswald-large-v3-turbo-m3) converted to whisper.cpp's ggml format (the same family as GGUF; whisper.cpp does not read `.gguf`).

| File | Size | Use |
|---|---|---|
| `ggml-oswald-m3-q5_0.bin` | 574 MB | **recommended**: smallest, no measurable accuracy loss |
| `ggml-oswald-m3-q8_0.bin` | 874 MB | |
| `ggml-oswald-m3-f16.bin` | 1,625 MB | reference; re-quantise from this |
| `ggml-silero-v6.2.0.bin` | 0.9 MB | voice-activity detector, **use it** (see below) |

## Accuracy

The same 304 held-out clips through every variant, greedy decoding. Error rates are on lowercase text without punctuation or accents, the form the model writes.

| Variant | Bible WER / CER | Radio WER / CER* | CMU WER / CER | Differs from PyTorch (CER) | Speed, RTX 4090 |
|---|---|---|---|---|---|
| original PyTorch fp16 | 4.37 / 1.47 | 16.74 / 5.40 | 3.80 / 0.59 | — | 118× real time |
| ggml f16 | 4.46 / 1.47 | 16.64 / 5.37 | 3.80 / 0.59 | 0.05% | 121× |
| ggml q8_0 | 4.40 / 1.47 | 16.55 / 5.37 | 2.53 / 0.30 | 0.11% | 135× |
| **ggml q5_0** | **4.22 / 1.43** | 16.47 / 5.41 | 2.53 / 0.30 | 0.27% | **140×** |

150 held-out Bible test clips, 150 radio test clips and the 4 held-out CMU clips (only 4, so one word moves that column). \*The radio references are machine transcripts, so that column is agreement rather than accuracy. Per-domain results for the original model are on its [model card](https://huggingface.co/jsbeaudry/oswald-large-v3-turbo-m3).

## Usage

```bash
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build && cmake --build build -j --config Release     # Metal on Apple Silicon, add -DGGML_CUDA=ON for NVIDIA
hf download jsbeaudry/oswald-large-v3-turbo-m3-ggml ggml-oswald-m3-q5_0.bin ggml-silero-v6.2.0.bin --local-dir models

ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav
./build/bin/whisper-cli -m models/ggml-oswald-m3-q5_0.bin -l ht \
  --vad -vm models/ggml-silero-v6.2.0.bin -f input.wav
```

- **`-l ht` is required.** whisper-cli defaults to English.
- **`--vad` matters for this model.** Tested on the converted q5_0:

  | Input | Without `--vad` | With `--vad` |
  |---|---|---|
  | two sentences separated by 6 s of silence | the second sentence is dropped | both sentences, with timestamps |
  | 8 s of silence | *"anpil moun ki te genyen moun ki te genyen…"* | nothing |

  A few silent training recordings carried transcripts, so the model can write text where nobody speaks. The detector sends it only speech.
- `whisper-server` serves the same model over HTTP with a small web page, for a browser front end on your own machine.
- The accuracy table used greedy decoding (`-bs 1 -bo 1`); whisper-cli's default is beam search with 5 beams, which is slower.

## Speed on a laptop

Apple M3 Pro, Metal, q5_0 with `--vad`: a 7–12 s clip takes 1.2–1.5 s including loading the model, about 9× faster than real time. The very first run after building takes about 25 s while Metal compiles its kernels.

## In a browser

The same `.bin` runs inside a browser through whisper.cpp's WebAssembly build (`examples/whisper.wasm`), with no server and no network once loaded. It runs on the CPU only, so expect it to be far slower than the native build.

## Output and limits

Lowercase Kreyòl without punctuation, numbers spelled out, modern spelling without apostrophes. French speech comes out as Kreyòl-looking words. Spontaneous, noisy or phone-quality audio transcribes worse than read speech. See the [original model card](https://huggingface.co/jsbeaudry/oswald-large-v3-turbo-m3) for the training data and its rights.

The VAD model is Silero VAD (MIT) as converted by [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad).
