# Cloud TTS for the Kreyòl kit

An image that runs the same `llama-tts` the local kit runs, behind an HTTP port, so the web page can
generate speech without the model on the machine.

## Why this image exists

The model was already deployed as a Hugging Face Inference Endpoint using their llama.cpp container.
That container cannot synthesise speech. Measured against the running endpoint:

```
POST /                          404   File Not Found
POST /v1/audio/speech           404   no such route
GET  /props                     200   build b11206, modalities {"vision":false,"video":false,"audio":true}
POST /v1/audio/transcriptions   400   "No input file found for transcription"
POST /completion                200   tokens [152074, 152932, 153780, ...]   content: ""
```

`audio: true` there means audio **in**, not out — the route it exposes is transcription. `/completion`
does emit the audio codec tokens, ids above Qwen3's 151,669-token text vocabulary, which is why
`content` comes back empty; but nothing in that container turns them into a waveform. Vocoding lives
in the separate `llama-tts` binary, and `llama-server` never runs it.

So this image runs that binary, kept loaded and fed one job per line by `llama-tts-serve.patch`,
exactly as `app/engine.py` does locally.

## The API

```
GET  /health              {"status": "ok"}         503 while loading, so a proxy waits
GET  /voices              {"voices": [...], "sample_rate": 24000}
POST /v1/audio/speech     -> audio/wav             {"input": "Bonjou", "voice": "kreyol_f1"}
POST /                    -> audio/wav             {"inputs": "Bonjou"}      the Hugging Face shape
```

`X-Audio-Seconds` and `X-Generate-Seconds` come back on every synthesis.

## Building it

Not on a laptop: the target is x86_64 CUDA and a development Mac is arm64, so a local build would run
under emulation for hours. `.github/workflows/cloud-tts.yml` builds it natively on a runner and pushes
to `ghcr.io/<owner>/kreyol-tts-cloud`.

```
gh workflow run cloud-tts.yml -f cuda_archs='75;89'
```

`cuda_archs` is the one knob worth touching: 75 is T4, 80 is A100, 86 is A10G, 89 is L4. Every extra
architecture is another full compile of the CUDA kernels, so build only the GPUs you will deploy on.

llama.cpp is pinned to `1af554f8fc78ba029665a47b839484d9763e2a75`, the commit the local kit was built
from. A different commit is a different voice — sampling defaults, the tokeniser and the codec path
all move between releases — and the point is to sound like the local kit, not merely to work. The
build fails rather than starting if the patch did not make it into the binary.

## Deploying it

The endpoint needs the **model repository** mounted (the image reads `/repository`, or `MODEL` and
`MMPROJ` if you set them) and the image pulled from GHCR. Make the GHCR package public, or give the
endpoint registry credentials.

Environment the image understands:

| | |
|---|---|
| `MODEL_DIR` | where to look for the GGUFs, default `/repository` |
| `MODEL` / `MMPROJ` | exact paths, if the search picks the wrong file |
| `NGL` | layers on the GPU, default 99 |
| `MAX_CHARS` | longest request, default 2000 |
| `PORT` | default 80 |

## Using it from the web page

Reglaj → *Ki kote vwa a fèt* → **Nan nyaj la**, then paste the address. The token is read from
`HF_TOKEN` or the `hf` CLI login by the local server, which adds the header itself; it is never sent
to the browser, never written to `travay/settings.json`, and never included in an error message.

Two behaviours worth knowing. A cloud failure falls back to the local model for that piece rather than
failing the reading. And a voice you added on your own machine is not in the image, so those readings
stay local — `/voices` lists what the service actually carries.
