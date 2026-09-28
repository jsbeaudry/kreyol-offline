"""Kreyòl speech on a GPU: llama-tts and whisper.cpp behind one RunPod worker.

Both engines stay loaded for the life of the worker. That is the whole point — loading them costs
eleven seconds, and a dialogue that pays that per turn is not a dialogue. The handler only moves bytes.

Two jobs:

    {"action": "tts", "text": "Bonjou", "voice": "kreyol_f1"}   -> {"audio": <base64 wav>, "seconds": 1.2}
    {"action": "stt", "audio": "<base64 wav>"}                  -> {"text": "bonjou"}

Audio crosses as base64 WAV because that is what RunPod's job payload carries. A three-second clip is
about 140 kB, which is nothing next to the time the GPU saves.
"""
from __future__ import annotations

import base64
import io
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import wave

import runpod

LLAMA_TTS = os.environ.get("LLAMA_TTS", "/opt/llama/bin/llama-tts")
WHISPER = os.environ.get("WHISPER", "/opt/whisper/bin/whisper-server")
MODELS = os.environ.get("MODEL_DIR", "/models")
VOICES = os.environ.get("VOICE_DIR", "/opt/voices")
ASR_PORT = int(os.environ.get("ASR_PORT", "8178"))
TTS_RATE, ASR_RATE = 24000, 16000
SAMPLING = ["-c", "2048", "-ngl", "99", "--temp", "0.9", "--top-k", "50", "--top-p", "1.0",
            "--repeat-penalty", "1.05"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kreyol_text import normalize                                      # noqa: E402


def find(pattern, suffix, exclude=None):
    """The first model matching, skipping names that contain `exclude`.

    The exclusion is the whole point: "mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf" contains "qwen3-tts" and
    sorts before the backbone, so searching for the backbone by name found the projector and llama-tts
    answered "CLIP cannot be used as main model". Same trap the other image documents and avoids.
    """
    for name in sorted(os.listdir(MODELS)):
        if exclude and exclude in name:
            continue
        if pattern in name and name.endswith(suffix):
            return os.path.join(MODELS, name)
    raise SystemExit(f"no {pattern}*{suffix} in {MODELS}"
                     + (f" (ignoring {exclude})" if exclude else ""))


class Voice:
    """One llama-tts, kept loaded, fed a job per line by llama-tts-serve.patch."""

    def __init__(self):
        self.proc, self.replies, self.lock = None, queue.Queue(), threading.Lock()

    def start(self):
        model = os.environ.get("TTS_MODEL") or find("qwen3-tts", ".gguf", exclude="mmproj")
        mmproj = os.environ.get("TTS_MMPROJ") or find("mmproj", ".gguf")
        print(f"starting: llama-tts -m {model} --mmproj {mmproj}", flush=True)
        # Its output goes to a file as well, so a failure can be quoted rather than guessed at. The
        # message below used to blame the patch for every failure, including this one, where the patch
        # was present and the model was wrong.
        log = open(TTS_LOG, "w")
        self.proc = subprocess.Popen(
            [LLAMA_TTS, "-m", model, "--mmproj", mmproj, *SAMPLING, "-p", "-", "-o", os.devnull],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        if self.replies.get(timeout=600) != "@@tts\tready":
            raise SystemExit(f"llama-tts never said ready: {tail(TTS_LOG)}")
        self.say("Bonjou.", os.path.join(VOICES, "kreyol_f1.wav"), "/tmp/warm.wav", 0)

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith("@@tts"):
                self.replies.put(line.rstrip("\n"))
        self.replies.put(None)

    def say(self, text, speaker, out, seed):
        with self.lock:
            self.proc.stdin.write(f"{out}\t{speaker}\t{seed}\t{' '.join(text.split())}\n")
            self.proc.stdin.flush()
            try:
                reply = self.replies.get(timeout=300)
            except queue.Empty:
                raise RuntimeError("llama-tts stopped answering")
        if reply is None or reply.split("\t")[1] != "ok":
            raise RuntimeError((reply or "no reply").split("\t")[-1])


WHISPER_LOG = "/tmp/whisper-server.log"
TTS_LOG = "/tmp/llama-tts.log"


def tail(path, lines=14):
    """The end of an engine's own output, for an error worth reading."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return " | ".join(line.strip() for line in f.read().splitlines()[-lines:] if line.strip())
    except OSError:
        return "(no output)"


def whisper_tail(lines=14):
    return tail(WHISPER_LOG, lines)


def start_whisper():
    model = os.environ.get("STT_MODEL") or find("ggml-oswald", ".bin")
    # No -t: the local kit runs whisper-server without one and works, while a serverless host reports
    # a hundred-odd vCPUs and "-t 128" is not a thing anyone tested. Match what is known to run.
    args = [WHISPER, "-m", model, "-l", "ht", "-bs", "1", "-bo", "1",
            "--host", "127.0.0.1", "--port", str(ASR_PORT)]
    print("starting: " + " ".join(args), flush=True)
    # Its output goes to a file as well as the console, so a failure can be quoted back in the error
    # rather than leaving "stopped while starting" as the whole story — which cost two rounds of this.
    log = open(WHISPER_LOG, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{ASR_PORT}/", timeout=2)
            print("whisper-server is answering", flush=True)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"whisper-server exited with {proc.returncode}: {whisper_tail()}")
            time.sleep(1)
    raise SystemExit(f"whisper-server did not answer in 180s: {whisper_tail()}")


def transcribe(wav_bytes):
    boundary = uuid.uuid4().hex
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0.0\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode(),
        wav_bytes, f"\r\n--{boundary}--\r\n".encode()])
    request = urllib.request.Request(
        f"http://127.0.0.1:{ASR_PORT}/inference", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    import json
    with urllib.request.urlopen(request, timeout=300) as response:
        return " ".join(json.loads(response.read()).get("text", "").split())


def trim_blip(frames):
    """The verbatim trim from app/engine.py, so cloud audio ends where local audio ends."""
    import numpy as np

    rate = TTS_RATE
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
            if runs and (start - runs[-1][1]) * n / rate < 0.12:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((start, i))
            start = None
    if not runs:
        return frames
    while len(runs) > 1:
        gap, length = (runs[-1][0] - runs[-2][1]) * n / rate, (runs[-1][1] - runs[-1][0]) * n / rate
        if gap < 0.10 or length > 0.15:
            break
        runs.pop()
    end = min(len(audio), runs[-1][1] * n + int(0.06 * rate))
    out = audio[:end].copy()
    fade = min(int(0.04 * rate), len(out))
    out[len(out) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def wav_of(frames, rate):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return out.getvalue()


VOICE = Voice()
# Loading is done on a thread so the SDK can register immediately. RunPod expects
# runpod.serverless.start() promptly and kills a worker that has not registered — and because it only
# streams a worker's logs once it has registered, a worker that dies loading prints nothing at all,
# which is how this failed three times with no output to read. Register first, load behind it, and make
# the first job wait.
_ready = threading.Event()
_failed = {"why": None}


def load():
    try:
        print("loading whisper.cpp and llama-tts…", flush=True)
        started = time.time()
        start_whisper()
        VOICE.start()
        print(f"both engines warm in {time.time() - started:.1f}s", flush=True)
        _ready.set()
    except BaseException as error:              # a load failure must reach a job, not vanish
        _failed["why"] = f"{type(error).__name__}: {error}"
        print(f"loading failed: {_failed['why']}", flush=True)
        _ready.set()


def wait_for_engines(timeout=600):
    if not _ready.wait(timeout):
        return "the speech models are still loading"
    return _failed["why"]


def handler(job):
    started = time.time()
    problem = wait_for_engines()
    if problem:
        return {"error": problem}
    data = job.get("input") or {}
    action = (data.get("action") or "tts").lower()
    try:
        if action == "stt":
            raw = base64.b64decode(data.get("audio") or "")
            if not raw:
                return {"error": "stt needs base64 wav in 'audio'"}
            return {"text": transcribe(raw), "took": round(time.time() - started, 3)}

        if action == "tts":
            text = normalize(data.get("text") or "")
            if not text.strip():
                return {"error": "tts needs 'text'"}
            name = os.path.basename(data.get("voice") or "kreyol_f1")
            speaker = os.path.join(VOICES, name if name.endswith(".wav") else f"{name}.wav")
            if not os.path.exists(speaker):
                return {"error": f"no voice {name}"}
            out = f"/tmp/{uuid.uuid4().hex}.wav"
            try:
                VOICE.say(text, speaker, out, int(data.get("seed") or time.time()))
                with wave.open(out, "rb") as w:
                    frames = trim_blip(w.readframes(w.getnframes()))
            finally:
                if os.path.exists(out):
                    os.remove(out)
            return {"audio": base64.b64encode(wav_of(frames, TTS_RATE)).decode(),
                    "seconds": round(len(frames) / 2 / TTS_RATE, 3),
                    "took": round(time.time() - started, 3)}

        return {"error": f"unknown action {action!r}; use tts or stt"}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print("registering with runpod while the models load", flush=True)
    runpod.serverless.start({"handler": handler})
