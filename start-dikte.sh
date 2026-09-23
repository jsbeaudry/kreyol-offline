#!/bin/bash
# Kreyòl dictation: hold Right Command, speak, release, and the text lands where your cursor is.
# Needs Accessibility and Microphone permission for whatever runs this (Terminal, iTerm, your editor).
# PORT=8178 ./start-dikte.sh shares the speech-to-text server the offline page already runs.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/dikte/dikte.py" ${PORT:+--port "$PORT"} "$@"
