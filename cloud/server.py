"""HTTP text-to-speech, the same llama-tts the local kit runs, behind a port.

Why this exists: the model was already deployed on a Hugging Face Inference Endpoint using their
llama.cpp container, and that container cannot synthesise. Its llama-server exposes /completion and
/v1/audio/transcriptions but no speech route, so a request returns the audio codec tokens — ids above
Qwen3's 151,669-token text vocabulary — with an empty `content` field and no way to turn them into a
waveform. Vocoding lives in the separate `llama-tts` binary, which the server image never runs.

So this image runs that binary instead, kept loaded and fed one job per line by
llama-tts-serve.patch, exactly as app/engine.py does locally. Same model, same sampling, same
trim_blip, so cloud audio matches what the local kit produces rather than merely resembling it.

    GET  /health                 -> {"status": "ok"}          (the endpoint's readiness probe)
    POST /v1/audio/speech        -> audio/wav                 {"input": "...", "voice": "kreyol_f1"}
    POST /                       -> audio/wav                 {"inputs": "..."}   (the HF shape)
    GET  /voices                 -> the reference voices this image carries
"""
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import wave

LLAMA_TTS = os.environ.get("LLAMA_TTS", "/opt/llama/bin/llama-tts")
REPO = os.environ.get("MODEL_DIR", "/repository")
VOICE_DIR = os.environ.get("VOICE_DIR", "/opt/voices")
TMP = os.environ.get("TMPDIR", "/tmp")
PORT = int(os.environ.get("PORT", "80"))
RATE = 24000
MAX_CHARS = int(os.environ.get("MAX_CHARS", "2000"))
# The sampling the local kit uses. Changing it changes the voice, so it is not a tuning knob.
SAMPLING = ["-c", "2048", "-ngl", os.environ.get("NGL", "99"), "--temp", "0.9", "--top-k", "50",
            "--top-p", "1.0", "--repeat-penalty", "1.05"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kreyol_text import normalize, split_text                          # noqa: E402


def pick(pattern, fallback=None):
    """The first file in the mounted repo matching a name fragment."""
    if not os.path.isdir(REPO):
        return fallback
    for name in sorted(os.listdir(REPO)):
        if pattern in name and name.endswith(".gguf"):
            return os.path.join(REPO, name)
    return fallback


def model_paths():
    """(backbone, projector). mmproj is matched first so it is not mistaken for the backbone."""
    mmproj = os.environ.get("MMPROJ") or pick("mmproj")
    backbone = os.environ.get("MODEL")
    if not backbone and os.path.isdir(REPO):
        for name in sorted(os.listdir(REPO)):
            if name.endswith(".gguf") and "mmproj" not in name:
                backbone = os.path.join(REPO, name)
                break
    return backbone, mmproj


class Voice:
    """One llama-tts kept loaded, one job at a time.

    The patched binary answers every job with a line starting "@@tts", so a reply is never confused
    with the model's own logging. One slot only: the GPU holds a single context and two concurrent
    jobs on one process would interleave their output files.
    """

    def __init__(self):
        self.proc, self.replies, self.lock = None, queue.Queue(), threading.Lock()
        self.ready = False

    def start(self):
        backbone, mmproj = model_paths()
        if not backbone:
            raise SystemExit(f"no .gguf backbone found in {REPO}; set MODEL")
        args = [LLAMA_TTS, "-m", backbone] + (["--mmproj", mmproj] if mmproj else []) + SAMPLING
        print(f"starting: {' '.join(args)}", flush=True)
        self.proc = subprocess.Popen(args + ["-p", "-", "-o", os.devnull], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready = self._reply(600) == "@@tts\tready"
        if not self.ready:
            raise SystemExit("llama-tts never said ready — is the serve patch in this build?")
        # The first job builds the embedding table and compiles kernels; pay it now, not on a request.
        try:
            self.say("Bonjou.", default_voice(), os.path.join(TMP, "warmup.wav"), 0)
            print("warmed up", flush=True)
        except RuntimeError as error:
            print(f"warm-up failed ({error}); serving anyway", flush=True)

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith("@@tts"):
                self.replies.put(line.rstrip("\n"))
        self.replies.put(None)

    def _reply(self, timeout):
        try:
            return self.replies.get(timeout=timeout)
        except queue.Empty:
            return None

    def say(self, text, speaker, out, seed):
        with self.lock:
            self.proc.stdin.write(f"{out}\t{speaker or ''}\t{seed}\t{' '.join(text.split())}\n")
            self.proc.stdin.flush()
            reply = self._reply(300)
        if reply is None:
            self.ready = False
            raise RuntimeError("the voice model stopped responding")
        fields = reply.split("\t")
        if fields[1] != "ok":
            raise RuntimeError(fields[-1])


def voices():
    return sorted(f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".wav")) \
        if os.path.isdir(VOICE_DIR) else []


def default_voice():
    have = voices()
    return os.path.join(VOICE_DIR, f"{have[0]}.wav") if have else ""


def voice_path(name):
    if not name:
        return default_voice()
    safe = os.path.basename(str(name))
    path = os.path.join(VOICE_DIR, safe if safe.endswith(".wav") else f"{safe}.wav")
    if not os.path.exists(path):
        raise ValueError(f"no voice called {safe!r}; this image carries {', '.join(voices())}")
    return path


def trim_blip(frames):
    """Cut the click the model leaves in the silence after a piece, and end the piece cleanly.

    Copied verbatim from app/engine.py, not reimplemented. A first attempt here approximated it with a
    peak threshold and no fade, which cut in different places — a cloud voice that clicks where the
    local one does not is a difference in the product, not in the deployment. If engine.py's version
    changes, this must change with it.
    """
    import numpy as np

    rate = RATE
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(0.010 * rate)
    if len(audio) < 4 * n:
        return frames
    energy = np.sqrt((audio[: len(audio) // n * n].reshape(-1, n) ** 2).mean(axis=1))
    loud = energy > max(0.006, 0.02 * float(energy.max()))
    runs, start = [], None
    for i, on in enumerate(np.append(loud, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if runs and (start - runs[-1][1]) * n / rate < 0.12:     # one burst often rings a second time
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((start, i))
            start = None
    if not runs:
        return frames
    while len(runs) > 1:      # drop every short burst that sits alone in the silence after the speech
        gap, length = (runs[-1][0] - runs[-2][1]) * n / rate, (runs[-1][1] - runs[-1][0]) * n / rate
        if gap < 0.10 or length > 0.15:
            break
        runs.pop()
    end = min(len(audio), runs[-1][1] * n + int(0.06 * rate))       # a little room after the last word
    out = audio[:end].copy()
    fade = min(int(0.04 * rate), len(out))       # fades away anything faint left in that room
    out[len(out) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != RATE or w.getnchannels() != 1:
            raise RuntimeError(f"{path}: expected {RATE} Hz mono, got "
                               f"{w.getframerate()} Hz / {w.getnchannels()}ch")
        return w.readframes(w.getnframes())


def as_wav(frames):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames)
    return out.getvalue()


def synthesise(voice, text, name, seed=None):
    """Text -> WAV bytes, chunked the way the local kit chunks it."""
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to read: send {\"input\": \"...\"}")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text is limited to {MAX_CHARS} characters, got {len(text)}")
    speaker = voice_path(name)
    seed = int(seed if seed is not None else time.time())
    gap = b"\x00\x00" * int(0.25 * RATE)
    pieces, started = [], time.time()
    for n, chunk in enumerate(split_text(normalize(text))):
        out = os.path.join(TMP, f"tts-{uuid.uuid4().hex}.wav")
        try:
            voice.say(chunk, speaker, out, seed + n)
            pieces.append(trim_blip(read_wav(out)))
        finally:
            if os.path.exists(out):
                os.remove(out)
    frames = gap.join(pieces)
    return as_wav(frames), len(frames) / 2 / RATE, time.time() - started


def serve(voice):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):                  # one tidy line, and never the auth header
            print(f"{self.command} {self.path} {fmt % args}", flush=True)

        def send(self, code, body, ctype="application/json"):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/healthz"):
                return self.send(200 if voice.ready else 503,
                                 {"status": "ok" if voice.ready else "loading"})
            if path == "/voices":
                return self.send(200, {"voices": voices(), "sample_rate": RATE})
            return self.send(404, {"error": f"no route {path}; try /health, /voices"})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/", "/v1/audio/speech", "/synthesize", "/generate"):
                return self.send(404, {"error": f"no route {path}; "
                                               "POST / or /v1/audio/speech"})
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
            except ValueError:
                return self.send(400, {"error": "body must be JSON"})
            if not isinstance(body, dict):
                return self.send(400, {"error": "body must be a JSON object"})
            # "input" is OpenAI's field, "inputs" is Hugging Face's; accept either.
            params = body.get("parameters") or {}
            text = body.get("input") or body.get("inputs") or body.get("text")
            name = body.get("voice") or params.get("voice")
            seed = body.get("seed", params.get("seed"))
            try:
                audio, seconds, took = synthesise(voice, text, name, seed)
            except ValueError as error:
                return self.send(400, {"error": str(error)})
            except RuntimeError as error:
                return self.send(503, {"error": str(error)})
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("X-Audio-Seconds", f"{seconds:.2f}")
            self.send_header("X-Generate-Seconds", f"{took:.2f}")
            self.end_headers()
            self.wfile.write(audio)

    print(f"listening on {PORT}; voices: {', '.join(voices()) or 'none'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    engine = Voice()
    engine.start()
    serve(engine)
