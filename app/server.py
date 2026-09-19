"""Kreyòl offline speech page: a local server for the two models in ~/kreyol-offline.

Speech to text goes through whisper-server, which keeps m3 loaded between requests. Text to speech goes
through one llama-tts that also stays loaded, reading jobs from stdin (llama-tts-serve.patch); with a stock
llama-tts, each reading starts its own process, about 1 s slower once warm. Everything binds to 127.0.0.1,
and nothing is sent anywhere else.

Usage: python3 app/server.py [--port 8177]
"""
import argparse
import atexit
import base64
import io
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kreyol-tts"))
from kreyol_text import normalize, split_text  # noqa: E402

WHISPER_SERVER = os.path.join(ROOT, "whisper.cpp/build/bin/whisper-server")
LLAMA_TTS = os.path.join(ROOT, "llama.cpp/build/bin/llama-tts")
ASR_MODEL = os.path.join(ROOT, "models/ggml-oswald-m3-q5_0.bin")
VAD_MODEL = os.path.join(ROOT, "models/ggml-silero-v6.2.0.bin")
VAD_TOOL = os.path.join(ROOT, "whisper.cpp/build/bin/whisper-vad-speech-segments")
TTS_MODEL = os.path.join(ROOT, "kreyol-tts/qwen3-tts-1.7b-kreyol-Q4_K_M.gguf")
TTS_MMPROJ = os.path.join(ROOT, "kreyol-tts/mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf")
VOICE_DIR = os.path.join(ROOT, "kreyol-tts/voices")
PAGE = os.path.join(ROOT, "app/index.html")
VOICES = [("kreyol_f1", "Fanm 1"), ("kreyol_f2", "Fanm 2"), ("kreyol_f3", "Fanm 3"),
          ("kreyol_m1", "Gason 1"), ("kreyol_v5", "Vwa 5")]
WHISPER_PORT = 8178      # set to the page's port + 1 in main(), so two copies never collide
MAX_BODY = 80 << 20      # an upload of a 10-minute recording fits comfortably
MAX_AUDIO_S = 600
MAX_TEXT = 2000
MERGE_GAP_S, MAX_REGION_S, PAD_S = 2.0, 28.0, 0.2   # pauses kept inside a line; Whisper reads 30 s windows
VOICE_S = (3.0, 15.0)    # a cloning reference shorter than 3 s is too little to go on; longer adds nothing

state = {"asr": "starting", "asr_error": "", "tts": "starting", "tts_mode": ""}
asr_lock, tts_lock = threading.Lock(), threading.Lock()
TMP = tempfile.mkdtemp(prefix="kreyol-page-")
custom_voices = {}
procs = []


class UserError(Exception):
    """A message already fit to show on the page."""


def to_wav(data, rate, max_s=None):
    """Any audio or video the browser sends (webm, mp4, m4a, mp3, wav) -> mono PCM WAV at `rate`."""
    src = os.path.join(TMP, f"in-{uuid.uuid4().hex}")
    dst = src + ".wav"
    with open(src, "wb") as f:
        f.write(data)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le"]
    cmd += ["-t", str(max_s)] if max_s else []
    r = subprocess.run(cmd + [dst], capture_output=True, text=True)
    os.remove(src)
    if r.returncode != 0 or not os.path.exists(dst):
        raise UserError("Odyo sa a pa ka li: eseye yon lòt fichye. (This audio could not be read; try another file.)")
    info = sf.info(dst)
    return dst, info.frames / info.samplerate


def start_whisper():
    """Load m3 once and keep it loaded. The first launch after a build compiles Metal kernels (~25 s)."""
    log = open(os.path.join(ROOT, "app/whisper-server.log"), "w")
    cmd = [WHISPER_SERVER, "-m", ASR_MODEL, "-l", "ht", "--vad", "-vm", VAD_MODEL, "-bs", "1", "-bo", "1",
           "--host", "127.0.0.1", "--port", str(WHISPER_PORT)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    procs.append(proc)
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            state.update(asr="failed", asr_error=f"whisper-server stopped (exit {proc.returncode}); see app/whisper-server.log")
            return
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{WHISPER_PORT}/", timeout=2)
            state["asr"] = "ready"
            print("speech to text ready", flush=True)
            return
        except Exception:
            time.sleep(0.5)
    state.update(asr="failed", asr_error="whisper-server did not start within 3 minutes")


def speech_regions(wav, total_s):
    """Where the speech is, from the VAD, merged across pauses under 2 s and cut to Whisper's 30 s window.

    These are the line timestamps. m3 was fine-tuned without timestamp tokens, so the segment times
    Whisper itself produces are wrong (it gave one line 0.1 s for ten words); the VAD reads the audio.
    """
    r = subprocess.run([VAD_TOOL, "-vm", VAD_MODEL, "-f", wav], capture_output=True, text=True, timeout=300)
    raw = [(float(a) / 100, float(b) / 100) for a, b in re.findall(r"start = ([\d.]+), end = ([\d.]+)", r.stdout)]
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= MERGE_GAP_S and b - merged[-1][0] <= MAX_REGION_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    regions = []
    for a, b in merged:            # one unbroken run longer than the window is cut into equal parts
        n = max(1, math.ceil((b - a) / MAX_REGION_S))
        regions += [(a + i * (b - a) / n, a + (i + 1) * (b - a) / n) for i in range(n)]
    return [(max(0.0, a - PAD_S), min(total_s, b + PAD_S)) for a, b in regions]


def whisper_text(wav_bytes):
    boundary = uuid.uuid4().hex
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0.0\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'.encode(),
        wav_bytes,
        f'\r\n--{boundary}--\r\n'.encode(),
    ])
    req = urllib.request.Request(f"http://127.0.0.1:{WHISPER_PORT}/inference", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return " ".join(json.load(resp).get("text", "").split())


def transcribe(data):
    if state["asr"] != "ready":
        raise UserError("Modèl tèks la poko pare. Tann yon ti moman. (Speech to text is still loading.)")
    wav, secs = to_wav(data, 16000)
    try:
        if secs > MAX_AUDIO_S:
            raise UserError(f"Odyo a twò long: {MAX_AUDIO_S // 60} minit maksimòm. (Audio is limited to {MAX_AUDIO_S // 60} minutes.)")
        with asr_lock:
            started = time.time()
            audio, sr = sf.read(wav, dtype="int16")
            segments = []
            for a, b in speech_regions(wav, secs):
                buf = io.BytesIO()
                sf.write(buf, audio[int(a * sr):int(b * sr)], sr, format="WAV", subtype="PCM_16")
                text = whisper_text(buf.getvalue())
                if text:
                    segments.append({"start": round(a, 2), "end": round(b, 2), "text": text})
            took = time.time() - started
    finally:
        os.remove(wav)
    return {"text": " ".join(s["text"] for s in segments), "segments": segments,
            "seconds": round(took, 2), "audio_seconds": round(secs, 2)}


def voice_path(voice):
    if voice in custom_voices:
        return custom_voices[voice]
    if voice in dict(VOICES):
        return os.path.join(VOICE_DIR, f"{voice}.wav")
    raise UserError("Chwazi yon vwa. (Choose a voice.)")


# -c 2048: llama-tts otherwise allocates a 32k-token KV cache (3.5 GB) for a few hundred tokens.
TTS_ARGS = ["-m", TTS_MODEL, "--mmproj", TTS_MMPROJ, "-c", "2048", "-ngl", "99",
            "--temp", "0.9", "--top-k", "50", "--top-p", "1.0", "--repeat-penalty", "1.05"]


class VoiceModel:
    """One llama-tts that stays loaded, fed a job per line (`-p -`, from llama-tts-serve.patch).

    Starting llama-tts costs about 1.1 s once the machine is warm (measured: a short sentence 2.0 s
    loaded vs 3.1 s as a fresh process), and several seconds more on the first run after a build, while
    Metal compiles its kernels; the warm-up job below pays that at startup instead of on the first click.
    A stock llama-tts has no job loop and never says "ready"; then every reading starts its own process
    and the page still works.
    """

    def __init__(self):
        self.proc, self.replies, self.loaded = None, None, False

    def start(self):
        log = open(os.path.join(ROOT, "app/llama-tts.log"), "w")
        # -o devnull: a stock build would read "-" as the text and write a file; make that harmless
        self.proc = subprocess.Popen([LLAMA_TTS, *TTS_ARGS, "-p", "-", "-o", os.devnull], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        procs.append(self.proc)
        self.replies = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.replies), daemon=True).start()
        self.loaded = self._reply(120) == "@@tts\tready"
        if self.loaded:   # the first job builds llama.cpp's embedding table; do it now, not on the first click
            try:
                self.run("Bonjou.", os.path.join(VOICE_DIR, VOICES[0][0] + ".wav"), os.path.join(TMP, "warmup.wav"), 0)
            except RuntimeError:
                self.loaded = False
        state["tts"], state["tts_mode"] = "ready", "loaded" if self.loaded else "per-reading"
        print(f"text to speech ready ({state['tts_mode']})", flush=True)

    @staticmethod
    def _pump(proc, replies):
        for line in proc.stdout:
            if line.startswith("@@tts"):
                replies.put(line.rstrip("\n"))
        replies.put(None)

    def _reply(self, timeout):
        try:
            return self.replies.get(timeout=timeout)
        except queue.Empty:
            return None

    def run(self, text, speaker, out, seed):
        # a job is one line: tabs and newlines inside the text would break it
        self.proc.stdin.write(f"{out}\t{speaker}\t{seed}\t{' '.join(text.split())}\n")
        self.proc.stdin.flush()
        reply = self._reply(300)
        if reply is None:
            self.loaded = False
            raise RuntimeError("the voice model stopped responding")
        fields = reply.split("\t")
        if fields[1] != "ok":
            raise RuntimeError(fields[-1])


def read_once(text, speaker, out, seed):
    """The fallback: a fresh llama-tts for one piece of text, paying the ~1 s start-up every time."""
    r = subprocess.run([LLAMA_TTS, *TTS_ARGS, "-p", text, "--tts-speaker-file", speaker, "-o", out,
                        "--seed", str(seed)], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(out):
        print(r.stderr[-2000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"llama-tts exited with {r.returncode}")


voice_model = VoiceModel()


def speak(text, voice):
    text = (text or "").strip()
    if not text:
        raise UserError("Ekri yon tèks an kreyòl anvan. (Write some Kreyòl text first.)")
    if len(text) > MAX_TEXT:
        raise UserError(f"Tèks la twò long: {MAX_TEXT} karaktè maksimòm. (Text is limited to {MAX_TEXT} characters.)")
    if state["tts"] != "ready":
        raise UserError("Modèl vwa a poko pare. Tann yon ti moman. (Text to speech is still loading.)")
    ref = voice_path(voice)
    chunks = split_text(normalize(text))
    pieces, rate = [], 24000
    with tts_lock:
        started = time.time()
        seed = int(time.time())
        for i, chunk in enumerate(chunks):
            out = os.path.join(TMP, f"tts-{uuid.uuid4().hex}.wav")
            if voice_model.loaded:
                try:
                    voice_model.run(chunk, ref, out, seed + i)
                except RuntimeError as e:
                    print(f"loaded voice model failed ({e}); reading this part the slow way", file=sys.stderr, flush=True)
                    if not voice_model.loaded:
                        state["tts_mode"] = "per-reading"
                    read_once(chunk, ref, out, seed + i)
            else:
                read_once(chunk, ref, out, seed + i)
            audio, rate = sf.read(out, dtype="float32")
            os.remove(out)
            pieces += [audio, np.zeros(int(0.25 * rate), dtype=np.float32)]
        took = time.time() - started
    audio = np.concatenate(pieces[:-1])
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return {"audio": base64.b64encode(buf.getvalue()).decode(), "spoken": " ".join(chunks), "parts": len(chunks),
            "seconds": round(took, 2), "audio_seconds": round(len(audio) / rate, 2)}


def add_voice(data):
    wav, secs = to_wav(data, 24000, max_s=VOICE_S[1])
    if secs < VOICE_S[0]:
        os.remove(wav)
        raise UserError(f"Klip la twò kout: omwen {VOICE_S[0]:.0f} segond. (The clip needs at least {VOICE_S[0]:.0f} seconds.)")
    vid = f"custom_{uuid.uuid4().hex[:8]}"
    custom_voices[vid] = wav
    return {"id": vid, "seconds": round(secs, 1)}


class Handler(BaseHTTPRequestHandler):
    server_version = "KreyolOffline/1"

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/health"):
            took = time.time() - getattr(self, "t0", time.time())
            print(f"{self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''} ({took:.1f} s)", flush=True)

    def send(self, status, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):   # readiness probes ask with HEAD
        self.t0 = time.time()
        self.send_response(200 if self.path.split("?")[0] in ("/", "/index.html", "/api/health") else 404)
        self.end_headers()

    def do_GET(self):
        self.t0 = time.time()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        if path == "/api/health":
            return self.send(200, {"asr": state["asr"], "asr_error": state["asr_error"],
                                   "tts": state["tts"], "tts_mode": state["tts_mode"],
                                   "voices": [{"id": v, "label": label} for v, label in VOICES]})
        m = re.fullmatch(r"/voices/([a-z0-9_]+)\.wav", path)
        if m:
            try:
                return self.send(200, open(voice_path(m.group(1)), "rb").read(), "audio/wav")
            except (UserError, OSError):
                return self.send(404, {"error": "voice not found"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        self.t0 = time.time()
        path = self.path.split("?")[0]
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            return self.send(413, {"error": "Fichye a twò gwo. (The file is too large.)"})
        data = self.rfile.read(size)
        try:
            if path == "/api/transcribe":
                return self.send(200, transcribe(data))
            if path == "/api/speak":
                req = json.loads(data or b"{}")
                return self.send(200, speak(req.get("text"), req.get("voice")))
            if path == "/api/voice":
                return self.send(200, add_voice(data))
            return self.send(404, {"error": "not found"})
        except UserError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:  # shown on the page, with the details in this terminal
            print(f"error on {path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return self.send(500, {"error": f"Yon erè rive: {e}. Gade tèminal la. (Something failed; see the terminal.)"})


def shutdown(*_):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    shutil.rmtree(TMP, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8177)
    args = parser.parse_args()
    global WHISPER_PORT
    WHISPER_PORT = args.port + 1
    missing = [p for p in (WHISPER_SERVER, VAD_TOOL, ASR_MODEL, VAD_MODEL, LLAMA_TTS, TTS_MODEL, TTS_MMPROJ) if not os.path.exists(p)]
    if missing:
        sys.exit("missing:\n  " + "\n  ".join(missing))
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required (brew install ffmpeg)")
    atexit.register(shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    threading.Thread(target=start_whisper, daemon=True).start()
    threading.Thread(target=voice_model.start, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True
    print(f"Kreyòl offline page: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
