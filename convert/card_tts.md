---
base_model: jsbeaudry/qwen3-tts-1.7b-kreyol
language:
  - ht
license: apache-2.0
library_name: gguf
pipeline_tag: text-to-speech
tags:
  - gguf
  - llama.cpp
  - qwen3-tts
  - haitian-creole
  - kreyol
  - voice-cloning
  - offline
---

# qwen3-tts-1.7b-kreyol GGUF

Haitian Creole (Kreyòl ayisyen) text-to-speech that runs offline with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-tts`: [`jsbeaudry/qwen3-tts-1.7b-kreyol`](https://huggingface.co/jsbeaudry/qwen3-tts-1.7b-kreyol), the voice-cloning variant, converted to GGUF. llama.cpp splits Qwen3-TTS into a **talker**, which predicts the audio codes, and an **mmproj**, which holds the speaker encoder and the decoder that turns codes into audio. You need one of each.

| File | Size | |
|---|---|---|
| `qwen3-tts-1.7b-kreyol-Q4_K_M.gguf` | 1,036 MB | talker, **recommended** |
| `qwen3-tts-1.7b-kreyol-Q8_0.gguf` | 1,848 MB | talker |
| `qwen3-tts-1.7b-kreyol-Q3_K_M.gguf` | 826 MB | talker, smallest that works |
| `qwen3-tts-1.7b-kreyol-f16.gguf` | 3,473 MB | talker, reference; re-quantise from this |
| `mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf` | 493 MB | mmproj, **recommended** |
| `mmproj-qwen3-tts-1.7b-kreyol-f16.gguf` | 701 MB | mmproj |
| `voices/kreyol_{f1,f2,f3,m1,v5}.wav` | ~0.4 MB each | reference clips for the five Kreyòl voices |
| `say.py`, `kreyol_text.py` | | text normalisation, chunking and synthesis in one command |

## Quality

Every sentence was synthesised, transcribed with [oswald-large-v3-turbo-m3](https://huggingface.co/jsbeaudry/oswald-large-v3-turbo-m3), and scored against the text it was asked to read: 36 Kreyòl sentences in two voices, 72 utterances per variant.

| Variant | CER | kreyol_f1 | kreyol_m1 |
|---|---|---|---|
| original PyTorch model | 3.28 | 2.90 | 3.66 |
| **Q4_K_M talker + Q8_0 mmproj** | **2.80** | 2.95 | 2.64 |

The 4-bit talker matches the original. Sampling is random, so these numbers move by roughly half a point between runs at this size; across three runs with different seeds and sentence sets, the original scored anywhere from 1.5% to 3.3%. On an RTX 4090 the GGUF generates at 0.41–0.44 of real time against 0.66 for PyTorch.

### How small the talker can go

A separate run on an Apple M3 Pro, judged with the **ggml** m3 through `whisper-server` rather than the
PyTorch m3 above, so its numbers belong to each other and not to the table above. Same 72 utterances.

| Variant | Size | CER | WER |
|---|---|---|---|
| Q4_K_M | 1,036 MB | 2.95 | 10.97 |
| Q3_K_M | 826 MB | 3.51 | 12.90 |
| Q2_K | 632 MB | — | unusable |

**Q3_K_M works.** It costs 0.56 CER for 210 MB, which is the same size as the run-to-run variation
described above — so read it as "not clearly worse and not clearly equal", not as a measured
degradation. Deciding properly would take several hundred utterances. Q4_K_M remains the recommendation;
Q3_K_M is there for a machine where 210 MB matters.

**Q2_K does not, and is not published.** Asked for a seven-second sentence it produced 41 seconds of
audio transcribing as `bot li li li li li ... gen怎么 li li ... дух solèy telesè`: degenerate looping,
foreign scripts, and no ability to stop. At 1.7B parameters two bits takes away both the model's
grounding and its stop token. One sentence settled it; the 72-utterance run was not needed. An IQ2
quantisation with an importance matrix built from Kreyòl calibration text might fare better, but that is
a project rather than a quantisation.

## Usage

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --config Release --target llama-tts     # Metal on Apple Silicon
hf download jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF --local-dir kreyol-tts \
  --include "*Q4_K_M.gguf" --include "mmproj-*Q8_0.gguf" --include "voices/*" --include "*.py"

pip install numpy soundfile
python kreyol-tts/say.py "Bonjou! Lekòl la ap louvri lendi a 8:30." -o bonjou.wav \
  --voice kreyol-tts/voices/kreyol_m1.wav --llama-tts build/bin/llama-tts
```

`say.py` spells out numbers and times the way the model was trained (`8:30` → *uit è trant*, `2026` → *de mil vennsis*), splits long text at sentence boundaries, and joins the pieces. A 30-second paragraph made this way transcribed back at 3.2% CER. To call `llama-tts` directly:

```bash
build/bin/llama-tts -m qwen3-tts-1.7b-kreyol-Q4_K_M.gguf --mmproj mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf \
  --tts-speaker-file voices/kreyol_f1.wav -c 2048 -ngl 99 \
  --temp 0.9 --top-k 50 --top-p 1.0 --repeat-penalty 1.05 \
  -p "Bonjou, kijan ou ye jodi a?" -o out.wav
```

- **`-c 2048`**: without it `llama-tts` allocates a 32k-token KV cache, 3.5 GB, for a few hundred tokens of speech.
- **Spell out digits first.** Without `say.py`, run the text through `kreyol_text.normalize`; the model was trained on words, not digits.
- One call stops after about 40 seconds of audio, which is why `say.py` splits the text.

## Speed on a laptop

Apple M3 Pro, Metal, Q4_K_M talker: speech is generated at 1.36× real time, or 1.15× including loading the model on every call (15.4 s of speech in 17.8 s). The first run after building is about 2.5× slower while Metal compiles its kernels. A text → speech → text round trip through the ggml ASR model on the same laptop came back nearly word for word.

## Voices

The base model clones a voice from a reference clip. The five clips in `voices/` were read by the five named voices of [qwen3-tts-1.7b-kreyol-voices](https://huggingface.co/jsbeaudry/qwen3-tts-1.7b-kreyol-voices) and chosen as the take the ASR model transcribed best (0–3.5% CER). llama.cpp cannot select a named speaker, so these clips are how those voices carry over: expect them close to the originals, not identical. Any clean 5–15 second Kreyòl recording works as a new voice.

## Language flag

For Qwen3-TTS, `llama-tts --tts-lang` defaults to English, while this model was fine-tuned with no language tag. Measured on the Q4_K_M talker, it makes no difference: 2.80% CER with the default, 2.92% with `--tts-lang fr`, and 2.77% with a patched build that reproduces the training prompt exactly. Stock llama.cpp is fine as it is.

## In a browser

No runtime runs Qwen3-TTS inside a browser today, in GGUF or any other format. For a browser interface that still works offline, serve a local page that calls `llama-tts` on the same machine.

## Limits

Sampling is random, so the same text reads slightly differently each time; change the seed to get another take. French words are read with Kreyòl pronunciation. See the [original model card](https://huggingface.co/jsbeaudry/qwen3-tts-1.7b-kreyol) for the training data and its rights.
