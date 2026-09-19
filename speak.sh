#!/bin/bash
# Kreyòl text to speech, offline. Voices: kreyol_f1 kreyol_f2 kreyol_f3 kreyol_m1 kreyol_v5
# Usage: ./speak.sh "Bonjou, kijan ou ye?" [voice] [out.wav]
set -eo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -n "$1" ] || { echo "usage: $0 \"text\" [voice] [out.wav]" >&2; exit 1; }
python3 "$DIR/kreyol-tts/say.py" "$1" -o "${3:-out.wav}" --voice "$DIR/kreyol-tts/voices/${2:-kreyol_f1}.wav" \
  --llama-tts "$DIR/llama.cpp/build/bin/llama-tts"
