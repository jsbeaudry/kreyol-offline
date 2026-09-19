#!/bin/bash
# Open the Kreyòl offline page: starts the local server, then the browser. Ctrl+C stops both models.
# PORT=8190 ./start-page.sh uses another port (the speech-to-text server takes PORT+1).
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-8177}
( sleep 2; open "http://127.0.0.1:$PORT" ) &
exec python3 "$DIR/app/server.py" --port "$PORT"
