#!/bin/bash
# Make this folder work offline: build whisper.cpp and llama.cpp (Metal on a Mac), apply the llama-tts
# patch, and download both models (2.1 GB; 2.5 GB in all when done). Running it again skips what is there.
set -eo pipefail
cd "$(dirname "$0")"

# The commits this was built and tested with. llama-tts-serve.patch is made against LLAMA_COMMIT.
WHISPER_COMMIT=5670d5c0bbcb148feabef84400a07cfca9aa3b30
LLAMA_COMMIT=1af554f8fc78ba029665a47b839484d9763e2a75
JOBS=$(sysctl -n hw.ncpu 2> /dev/null || nproc)
LOG=setup.log
: > "$LOG"

for tool in git cmake ffmpeg hf python3; do
  command -v "$tool" > /dev/null || { echo "missing: $tool (see README.md)" >&2; exit 1; }
done
python3 -c "import numpy, soundfile" 2> /dev/null ||
  { echo "python3 needs numpy and soundfile: pip3 install numpy soundfile" >&2; exit 1; }

quiet() {  # run a command with its output in $LOG; on failure show the end of it
  "$@" >> "$LOG" 2>&1 || { tail -25 "$LOG" >&2; echo "failed: $* (full log in $LOG)" >&2; exit 1; }
}

fetch() {  # fetch <name> <commit>: that one commit of github.com/ggml-org/<name>, without history
  [ -d "$1/.git" ] && git -C "$1" rev-parse -q --verify HEAD > /dev/null 2>&1 && return
  quiet git init -q "$1"
  quiet git -C "$1" fetch -q --depth 1 "https://github.com/ggml-org/$1" "$2"
  quiet git -C "$1" checkout -q FETCH_HEAD
}

have() {  # have <file>...: all of them exist
  for f in "$@"; do [ -f "$f" ] || return 1; done
}

download_failed() {
  echo "download failed. The model repos are private: run 'hf auth login' with an account that can read them." >&2
  exit 1
}

echo "whisper.cpp (speech to text)..."
fetch whisper.cpp "$WHISPER_COMMIT"
quiet cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
quiet cmake --build whisper.cpp/build -j "$JOBS" --target whisper-cli whisper-server whisper-vad-speech-segments

echo "llama.cpp (text to speech)..."
fetch llama.cpp "$LLAMA_COMMIT"
if git -C llama.cpp apply --check ../llama-tts-serve.patch 2> /dev/null; then
  git -C llama.cpp apply ../llama-tts-serve.patch
elif ! git -C llama.cpp apply -R --check ../llama-tts-serve.patch 2> /dev/null; then
  echo "  llama-tts-serve.patch does not apply here; the page will start llama-tts once per reading instead"
fi
quiet cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
quiet cmake --build llama.cpp/build -j "$JOBS" --target llama-tts

echo "models..."
have models/ggml-oswald-m3-q5_0.bin models/ggml-silero-v6.2.0.bin ||
  hf download jsbeaudry/oswald-large-v3-turbo-m3-ggml ggml-oswald-m3-q5_0.bin ggml-silero-v6.2.0.bin \
    --local-dir models || download_failed
have kreyol-tts/qwen3-tts-1.7b-kreyol-Q4_K_M.gguf kreyol-tts/mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf \
     kreyol-tts/say.py kreyol-tts/kreyol_text.py kreyol-tts/voices/kreyol_{f1,f2,f3,m1,v5}.wav ||
  hf download jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF --local-dir kreyol-tts \
    --include "*Q4_K_M.gguf" --include "mmproj-*Q8_0.gguf" --include "voices/*" --include "*.py" ||
    download_failed

echo "Ready. ./start-page.sh opens the page; ./transcribe.sh and ./speak.sh work from the command line."
