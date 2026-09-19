#!/bin/bash
# Kreyòl speech to text, offline: any audio file ffmpeg can read -> text on stdout.
# Usage: ./transcribe.sh recording.m4a
set -eo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$1" ] || { echo "usage: $0 <audio file>" >&2; exit 1; }
TMP=$(mktemp -t kreyol).wav; trap 'rm -f "$TMP"' EXIT
ffmpeg -loglevel error -y -i "$1" -ar 16000 -ac 1 -c:a pcm_s16le "$TMP"
# -l ht: whisper-cli defaults to English. --vad: without it the model drops speech after long pauses
# and writes text over silence.
"$DIR/whisper.cpp/build/bin/whisper-cli" -m "$DIR/models/ggml-oswald-m3-q5_0.bin" -l ht \
  --vad -vm "$DIR/models/ggml-silero-v6.2.0.bin" -nt -np -f "$TMP" 2>/dev/null | sed 's/^ *//' | grep -v '^$'
