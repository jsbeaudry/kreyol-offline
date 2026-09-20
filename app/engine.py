"""The two models behind the page, and the audio plumbing they share.

Speech to text: whisper-server keeps m3 loaded. Silero VAD finds the speech and each region is transcribed on
its own: m3 was fine-tuned without timestamp tokens, so the segment times Whisper itself produces are wrong
(it gave one line 0.1 s for ten words), while the VAD reads the audio.

Text to speech: one llama-tts stays loaded and reads jobs from stdin (llama-tts-serve.patch). With a stock
llama-tts, every piece of text starts its own process, about 1 s slower once warm.

Everything on the page shares the two models. The locks are taken per speech region and per piece of text,
so a quick request waits for at most one piece of a long job, never for the whole job.
"""
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kreyol-tts"))
from kreyol_text import normalize as tts_normalize, split_text  # noqa: E402

WHISPER_SERVER = os.path.join(ROOT, "whisper.cpp/build/bin/whisper-server")
LLAMA_TTS = os.path.join(ROOT, "llama.cpp/build/bin/llama-tts")
ASR_MODEL = os.path.join(ROOT, "models/ggml-oswald-m3-q5_0.bin")
VAD_MODEL = os.path.join(ROOT, "models/ggml-silero-v6.2.0.bin")
VAD_TOOL = os.path.join(ROOT, "whisper.cpp/build/bin/whisper-vad-speech-segments")
TTS_MODEL = os.path.join(ROOT, "kreyol-tts/qwen3-tts-1.7b-kreyol-Q4_K_M.gguf")
TTS_MMPROJ = os.path.join(ROOT, "kreyol-tts/mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf")
VOICE_DIR = os.path.join(ROOT, "kreyol-tts/voices")
REQUIRED = (WHISPER_SERVER, VAD_TOOL, ASR_MODEL, VAD_MODEL, LLAMA_TTS, TTS_MODEL, TTS_MMPROJ)

# Median pitch of the reference clips: f1 198 Hz, f2 240, f3 208 (women); m1 113, v5 118 (men).
VOICES = [("kreyol_f1", "Fanm 1"), ("kreyol_f2", "Fanm 2"), ("kreyol_f3", "Fanm 3"),
          ("kreyol_m1", "Gason 1"), ("kreyol_v5", "Vwa 5")]
ASR_RATE, TTS_RATE = 16000, 24000
MERGE_GAP_S, MAX_REGION_S, PAD_S = 2.0, 28.0, 0.2   # pauses kept inside a line; Whisper reads 30 s windows
VOICE_S = (3.0, 15.0)    # a cloning reference shorter than 3 s is too little to go on; longer adds nothing
WHISPER_PORT = 8178      # server.main() sets it to the page's port + 1, so two copies never collide

state = {"asr": "starting", "asr_error": "", "tts": "starting", "tts_mode": ""}
asr_lock, tts_lock = threading.Lock(), threading.Lock()
TMP = tempfile.mkdtemp(prefix="kreyol-page-")
custom_voices = {}       # voice id -> reference WAV, for this session only
procs = []


class UserError(Exception):
    """A message already fit to show on the page, in Kreyòl with the English in parentheses."""


def require(kind):
    """Refuse politely while a model is loading or if it failed."""
    st = state[kind]
    if st == "ready":
        return
    name = ("Modèl tèks la", "Speech to text") if kind == "asr" else ("Modèl vwa a", "Text to speech")
    if st == "starting":
        raise UserError(f"{name[0]} poko pare. Tann yon ti moman. ({name[1]} is still loading.)")
    raise UserError(f"{name[0]} pa demare. ({name[1]} failed to start: {state.get(kind + '_error') or 'see the terminal'}.)")


def wait_ready(kind, stopped=lambda: False):
    """For background jobs: block until the model is ready."""
    while state[kind] == "starting" and not stopped():
        time.sleep(1)
    if not stopped():        # a stopped job reports "stopped", not "still loading"
        require(kind)


# ---------- audio files ----------

def convert(src, dst, rate, max_s=None):
    """Any audio or video file ffmpeg reads -> mono 16-bit WAV at `rate`. Returns its length in seconds."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn", "-ar", str(rate), "-ac", "1",
           "-c:a", "pcm_s16le"]
    cmd += ["-t", str(max_s)] if max_s else []
    r = subprocess.run(cmd + [dst], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        if os.path.exists(dst):
            os.remove(dst)
        raise UserError("Odyo sa a pa ka li: eseye yon lòt fichye. (This audio could not be read; try another file.)")
    info = sf.info(dst)
    return info.frames / info.samplerate


def to_wav(data, rate, max_s=None):
    """Bytes the browser sent (webm, mp4, m4a, mp3, wav...) -> a temporary WAV path and its length."""
    src = os.path.join(TMP, f"in-{uuid.uuid4().hex}")
    with open(src, "wb") as f:
        f.write(data)
    try:
        dst = src + ".wav"
        return dst, convert(src, dst, rate, max_s)
    finally:
        os.remove(src)


def wav_bytes(audio, rate):
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def region_wav(wav, a, b):
    """Seconds a..b of a WAV file as WAV bytes, read without loading the whole file."""
    with sf.SoundFile(wav) as f:
        sr = f.samplerate
        f.seek(min(f.frames, int(a * sr)))
        data = f.read(max(0, int((b - a) * sr)), dtype="int16")
    return wav_bytes(data, sr)


# ffmpeg output options per format; the names and defaults follow OpenAI's speech API.
FORMATS = {
    "mp3": (["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"], "audio/mpeg"),
    "opus": (["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"], "audio/ogg"),
    "aac": (["-c:a", "aac", "-b:a", "64k", "-f", "adts"], "audio/aac"),
    "flac": (["-c:a", "flac", "-f", "flac"], "audio/flac"),
    "wav": (["-c:a", "pcm_s16le", "-f", "wav"], "audio/wav"),
    "pcm": (["-c:a", "pcm_s16le", "-f", "s16le"], "audio/pcm"),   # raw 24 kHz 16-bit mono, like OpenAI's
}


def tempo_filter(speed):
    """atempo takes 0.5-100 per filter; chain two for slower speeds."""
    steps = []
    while speed < 0.5:
        steps.append(0.5)
        speed /= 0.5
    steps.append(speed)
    return ",".join(f"atempo={s:.4f}" for s in steps)


def encode(audio, fmt, speed=1.0, rate=TTS_RATE):
    """float32 samples -> bytes in one of FORMATS.

    ffmpeg writes to a file, not a pipe: FLAC and WAV record the length in a header that ffmpeg can only
    fill in by seeking back, and a piped FLAC reported a length of billions of seconds.
    """
    wav = wav_bytes(audio, rate)
    if fmt == "wav" and speed == 1.0:
        return wav
    args, _ = FORMATS[fmt]
    filt = ["-filter:a", tempo_filter(speed)] if speed != 1.0 else []
    out = os.path.join(TMP, f"enc-{uuid.uuid4().hex}.{fmt}")
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "wav", "-i", "pipe:0", *filt, *args, out],
                           input=wav, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg could not make {fmt}: {r.stderr.decode(errors='replace')[-300:]}")
        with open(out, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out):
            os.remove(out)


def encode_file(src, dst, fmt):
    """A WAV on disk -> another format on disk (for long outputs that should not sit in memory)."""
    args, _ = FORMATS[fmt]
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, *args, dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not make {fmt}: {r.stderr[-300:]}")


# ---------- speech to text ----------

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
    """Where the speech is, from the VAD, merged across pauses of up to 2 s and cut to Whisper's 30 s window.

    These are the line timestamps (see the module docstring for why Whisper's own are not used).
    """
    r = subprocess.run([VAD_TOOL, "-vm", VAD_MODEL, "-f", wav], capture_output=True, text=True, timeout=1800)
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
    return [(round(max(0.0, a - PAD_S), 2), round(min(total_s, b + PAD_S), 2)) for a, b in regions]


def whisper_text(wav_data):
    boundary = uuid.uuid4().hex
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0.0\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'.encode(),
        wav_data,
        f'\r\n--{boundary}--\r\n'.encode(),
    ])
    req = urllib.request.Request(f"http://127.0.0.1:{WHISPER_PORT}/inference", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return " ".join(json.loads(resp.read()).get("text", "").split())


def transcribe_region(wav, a, b):
    data = region_wav(wav, a, b)
    with asr_lock:
        return whisper_text(data)


def transcribe_bytes(data, max_s):
    """One request, start to finish: the Koute tool, reading practice and the API use this."""
    require("asr")
    wav, secs = to_wav(data, ASR_RATE)
    try:
        if max_s and secs > max_s:
            minutes = max_s // 60
            raise UserError(f"Odyo a twò long: {minutes} minit maksimòm. Pou pi long, sèvi ak Transkripsyon. "
                            f"(Audio is limited to {minutes} minutes here; use Transkripsyon for longer recordings.)")
        started = time.time()
        segments = []
        for a, b in speech_regions(wav, secs):
            text = transcribe_region(wav, a, b)
            if text:
                segments.append({"start": a, "end": b, "text": text})
        took = time.time() - started
    finally:
        os.remove(wav)
    return {"text": " ".join(s["text"] for s in segments), "segments": segments,
            "seconds": round(took, 2), "audio_seconds": round(secs, 2)}


# ---------- text to speech ----------

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


def voice_path(voice):
    if voice in custom_voices:
        return custom_voices[voice]
    if voice in dict(VOICES):
        return os.path.join(VOICE_DIR, f"{voice}.wav")
    raise UserError("Chwazi yon vwa. (Choose a voice.)")


def trim_blip(audio, rate=TTS_RATE):
    """Cut the click the model sometimes leaves in the silence after a piece, and end the piece cleanly.

    Measured over 19 readings: 11 ended with a 10-30 ms burst, 0.15-1.2 s after the last word and sometimes as
    loud as the speech itself. In a reading made of several pieces you hear one at every join and one at the
    end. A run that short, that far behind the speech, is never a word.
    """
    n = int(0.010 * rate)
    if len(audio) < 4 * n:
        return audio
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
        return audio
    while len(runs) > 1:      # drop every short burst that sits alone in the silence after the speech
        gap, length = (runs[-1][0] - runs[-2][1]) * n / rate, (runs[-1][1] - runs[-1][0]) * n / rate
        if gap < 0.10 or length > 0.15:
            break
        runs.pop()
    end = min(len(audio), runs[-1][1] * n + int(0.06 * rate))       # a little room after the last word
    out = audio[:end].copy()
    fade = min(int(0.04 * rate), len(out))       # fades away anything faint left in that room
    out[len(out) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def synth_chunk(text, ref, seed):
    """One piece of text (at most ~220 characters) in the voice of `ref` -> float32 samples at 24 kHz."""
    out = os.path.join(TMP, f"tts-{uuid.uuid4().hex}.wav")
    with tts_lock:
        if voice_model.loaded:
            try:
                voice_model.run(text, ref, out, seed)
            except RuntimeError as e:
                print(f"loaded voice model failed ({e}); reading this part the slow way", file=sys.stderr, flush=True)
                if not voice_model.loaded:
                    state["tts_mode"] = "per-reading"
                read_once(text, ref, out, seed)
        else:
            read_once(text, ref, out, seed)
    audio, _ = sf.read(out, dtype="float32")
    os.remove(out)
    return trim_blip(audio)


def silence(seconds):
    return np.zeros(int(seconds * TTS_RATE), dtype=np.float32)


def speak(text, voice, max_chars):
    """Text -> (float32 audio at 24 kHz, the pieces as spoken, seconds of work)."""
    text = (text or "").strip()
    if not text:
        raise UserError("Ekri yon tèks an kreyòl anvan. (Write some Kreyòl text first.)")
    if len(text) > max_chars:
        raise UserError(f"Tèks la twò long: {max_chars} karaktè maksimòm. Pou pi long, sèvi ak Dokiman. "
                        f"(Text is limited to {max_chars} characters here; use Dokiman for longer text.)")
    require("tts")
    ref = voice_path(voice)
    chunks = split_text(tts_normalize(text))
    started, seed = time.time(), int(time.time())
    pieces = []
    for i, chunk in enumerate(chunks):
        pieces += [synth_chunk(chunk, ref, seed + i), silence(0.25)]
    return np.concatenate(pieces[:-1]), chunks, time.time() - started


def add_voice(data):
    wav, secs = to_wav(data, TTS_RATE, max_s=VOICE_S[1])
    if secs < VOICE_S[0]:
        os.remove(wav)
        raise UserError(f"Klip la twò kout: omwen {VOICE_S[0]:.0f} segond. (The clip needs at least {VOICE_S[0]:.0f} seconds.)")
    vid = f"custom_{uuid.uuid4().hex[:8]}"
    custom_voices[vid] = wav
    return {"id": vid, "seconds": round(secs, 1)}


# ---------- lifecycle ----------

def start():
    threading.Thread(target=start_whisper, daemon=True).start()
    threading.Thread(target=voice_model.start, daemon=True).start()


def shutdown():
    for p in procs:
        if p.poll() is None:
            p.terminate()
    shutil.rmtree(TMP, ignore_errors=True)
