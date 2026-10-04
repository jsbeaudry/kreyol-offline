# Klara mobile — Kreyòl speech on the phone (Android)

A React Native (Expo 57) speech demo that runs **entirely on the phone**, no internet:

| | model | runtime | size |
|---|---|---|---|
| **Tande** (hold to talk) | Qwen3-ASR 0.6B Kreyòl v2, GGUF Q4_K_M + mmproj Q8_0 | `llama.rn` (llama.cpp) | 692 MB |
| **Pale** (type → speech) | Pocket TTS Kreyòl v1, ONNX: append-only KV LM in 4 bits, INT8 elsewhere, 6 voices | `onnxruntime-react-native` | 240 MB |

The models download on first launch (893 MB from two public Hugging Face repos,
[jsbeaudry/qwen3-asr-0.6b-kreyol-v2-GGUF](https://huggingface.co/jsbeaudry/qwen3-asr-0.6b-kreyol-v2-GGUF) and
[jsbeaudry/pocket-tts-kreyol-onnx](https://huggingface.co/jsbeaudry/pocket-tts-kreyol-onnx), ~2 min on Wi-Fi);
after that nothing needs the internet.

On a Galaxy A16 (Exynos 1330), the ASR transcribed a test clip exactly at ~0.9× real time. Pocket's
first sound came after 1.2–2 s and playback had no gaps, at RTF 1.3–1.45. See "Speed on a real phone" below.

## Layout

- `klara-mobile/` — the app. `App.tsx` (screen), `src/asr.ts` (llama.rn, "Transcribe audio to text",
  noise filter), `src/pocket.ts` (Pocket ONNX loop: text conditioner → LM + flow step per 80 ms frame
  → streaming Mimi decode), `src/tokenizer.ts` (Unigram tokenizer, identical ids to Python on 84/84
  texts), `src/kreyolText.ts` (numbers/times/money spelled out, identical to Klara's on 12/12),
  `src/audio.ts` (16 kHz mic → WAV, 24 kHz streaming playback), `src/models.ts` (download / location),
  `plugins/withOnnxruntimePackage.js` (registers ORT's package; arm64-only; 2 llama.rn variants).
- `models/` (not in git) — what goes on the phone: `asr/` and `pocket/`, the same files as the two HF repos.
- `export/` — how the Pocket bundle was made (not needed to build the app):
  - `lomotron.patch` — applies to [lomotron/pocket-tts-onnx-export](https://github.com/lomotron/pocket-tts-onnx-export)
    at `8f52199` (pocket-tts 3.3.0 architecture). It lets the exporter take a local `--config` and
    `POCKET_SEQ_LEN`. Clone that repo into `export/lomotron/` and apply the patch.
  - `export_flow_lm_kv.py` — the append-only KV LM export used on phones. Copy it to
    `export/lomotron/scripts/`, then run it from `export/lomotron` with `PYTHONPATH=.`.
    `q4.py` makes the 4-bit file from its output.
  - `cfg/kreyol_v1_512.yaml` — our fine-tune; the tokenizer line is a placeholder (graphs take ids).
  - `make_voices.py` — voice embeddings made in PyTorch exactly as pocket-tts conditions on a clip
    (Mimi → speaker_proj; BOS prepended at runtime). The ONNX runtime's own audio path scored worse
    (CER 14.8 vs ~10), which is why voices ship as embeddings.
  - `validate.py`, `validate_kv.py`, `score.py` — PyTorch vs ONNX on 20 held-out sentences, scored
    with Qwen-ASR CER. Run `validate_kv.py` on the export bundle that has `bos_before_voice.npy`:
    without it the voice gets no BOS and CER jumps to ~80%.

## Build and run (Android)

```bash
cd klara-mobile && npm install
KLARA_DEBUGGABLE=1 npx expo prebuild --platform android   # debuggable: lets push-models.sh copy models in
cd android && ./gradlew assembleRelease                       # JAVA_HOME = JDK 17, ANDROID_HOME set
adb install -r app/build/outputs/apk/release/app-release.apk
../../push-models.sh                                          # optional: copy models in over USB
```

The app downloads any model it is missing on first launch (`src/models.ts`). `push-models.sh` only
skips that download on a development phone; it needs the debuggable build.

## Gotchas met

- `onnxruntime-react-native` 1.24's Gradle file uses `VersionNumber`, gone in Gradle 9 → `patches/`.
- It is not autolinked under Expo 57 → `NativeModules.Onnxruntime` was null; the config plugin adds it.
- `llama.rn` 0.13 rc forces a from-source llama.cpp build; all six CPU variants filled the disk →
  two variants (`rnllama;rnllama_v8_2_dotprod`), arm64 only.
- expo-file-system `File` needs `file://` URIs; ORT and llama.rn take plain paths.
- react-native-audio-api's queue source `start()` rejects its own default offset → `start(t, 0)`.
- A folder created with adb under `Android/data/<pkg>` before install is not readable by the app.
- iOS is untested: Xcode is not installed on this Mac.

## Speed on a real phone (Galaxy A16, Exynos 1330, 2026-10-04)

The stock Pocket export was ~5× too slow on the phone (RTF 5.4). Each 80 ms frame moved the LM's
fixed 100 MB KV cache in and out (a `ScatterND` copy per layer, plus the binding's output copy).
Changes:

- **Append-only LM export:** `export/export_flow_lm_kv.py`. Each layer takes past
  K/V at their used length and returns only the new positions; JS appends them (`KVCache` in
  `src/pocket.ts`). It matches the stock streaming path exactly in PyTorch, and ONNX is within 2e-6.
- **4-bit LM weights:** `MatMulNBits`, block 32, `accuracy_level=4` (`export/q4.py`).
  - 190 MB instead of 304 MB.
  - CER over 20 held-out sentences × 2 seeds (`export/validate_kv.py` + `score.py`): 10.7% vs 9.8% for INT8, within noise.
  - `bundle.json` `lm_file` picks the file.
- **ORT threads = 2:** 1, 3 and 4 were slower. The A55 cores slow ORT down, and pinning didn't help.
- **Mimi:** its 39 pass-through state outputs are never fetched, and the big state outputs are
  written in place through preallocated fetches. The binding was patched to accept `extra.session`
  config entries.
- **Decoding stays sequential:** running Mimi in parallel with the LM doubled the LM's frame time
  (memory bandwidth).
- **Per-sentence pre-buffer:** each chunk starts playing once the audio in hand outlasts the rest
  of that chunk at the measured rate. A slow phone pauses between sentences rather than stuttering.

| | before | after |
|---|---|---|
| LM per frame | 311 ms | 43 ms alone, 60–70 ms in the loop |
| RTF (sample text) | 5.4 | 1.3–1.45 (big cores thermally capped to 1.73 GHz) |
| first sound | 7.9 s, then a cut | 1.2–2.0 s, no gaps |

**ASR on the phone:** exact transcript on the CMU test clip; ~6 s for 6.9 s of audio (cold). An
identical re-run is 1.7 s only because llama.rn caches the prompt. More than 4 threads is ~4× slower.

**Dev hooks (debuggable build):** long-press "Tande" to transcribe `cache/test-heard.wav`;
long-press "Pale" for the LM thread benchmark; long-press "Klara" for the ASR thread benchmark.
