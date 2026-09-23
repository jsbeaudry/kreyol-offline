"""Read the selected text aloud with the Kreyòl voice, starting before the whole thing is made.

Synthesis runs at about 1.1 to 1.3 times real time, so a paragraph made in one piece would leave you
waiting most of its length in silence. It is, however, comfortably faster than playback, so the work is
split: a short first block starts the sound quickly, and the rest is made while that block plays. The
first block is deliberately smaller than the others for that reason alone.

Measured on an M3 Pro: loading llama-tts and its handshake take 8.3 s, which is why `preload` runs in
the background when the service starts rather than on the first selection. After that a 53-character
sentence takes 6.05 s to produce 6.6 s of audio.

The job protocol is the one app/engine.py uses, from llama-tts-serve.patch: one tab-separated line in,
one `@@tts` line back. Without the patch llama-tts has no job loop, so `preload` reports that and
reading stays unavailable rather than silently starting a process per block.
"""
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np

import settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_TTS = os.path.join(ROOT, 'llama.cpp/build/bin/llama-tts')
VOICE_DIR = os.path.join(ROOT, 'kreyol-tts/voices')
# -c 2048 for the same reason engine.py gives: otherwise a 32k KV cache costs 3.5 GB for a few hundred
# tokens. The sampling settings match the page so a sentence sounds the same in both.
SAMPLING = ['-c', '2048', '-ngl', '99', '--temp', '0.9', '--top-k', '50', '--top-p', '1.0',
            '--repeat-penalty', '1.05']


def tts_args():
    """Built when the model starts, not at import, so a change in the menu is picked up."""
    return ['-m', settings.get('tts_model'), '--mmproj', settings.get('tts_mmproj'), *SAMPLING]

FIRST_BLOCK = 45         # small, so the sound starts sooner
BLOCK = 220              # what kreyol_text.split_text uses by default
MAX_CHARS = 5000         # a whole document selected by accident should not become a ten-minute reading
MIN_BLOCK = 25           # shorter than this, the fixed cost of a call outweighs the audio it makes
# Trimming the click also takes the trailing silence, and that silence had been the cover under
# which the next block was made. Measured after trimming, making a block costs about 0.93 of its own
# playing time, so a block can only be about 1.15 times the one before it. Blocks split at sentence
# ends, so what is left of the shortfall lands between sentences, where a pause belongs.
GROWTH = 1.15
PAUSE = 0.3              # between blocks: a breath between sentences, and cover for the next one
STEP = 0.08              # how often the level is reported while a block plays
TTS_RATE = 24000         # what llama-tts writes

sys.path.insert(0, os.path.join(ROOT, 'kreyol-tts'))


def blocks(text):
    """Split into pieces that start small and grow, so the sound starts soon and then never stops.

    A short first piece is heard quickly; each following one may be up to GROWTH times the last,
    which is the most that can be made during the previous piece's playback.
    """
    import kreyol_text
    text = kreyol_text.normalize(' '.join(text.split()))[:MAX_CHARS]
    if not text:
        return []
    # One sentence at a time. split_text groups sentences up to a limit rather than separating them,
    # so asking it for small pieces cuts inside sentences (a ten-character block cost 2.7 s to make
    # for 1.9 s of audio) and asking it for large ones hands back the whole paragraph as one piece
    # (169 characters, eleven seconds before a sound). Its own sentence rule, used directly, gives
    # units that are neither.
    units = []
    for sentence in (s for s in re.split(r'(?<=[.!?…])\s+', text) if s):
        units.extend(kreyol_text.split_text(sentence, max_chars=BLOCK)
                     if len(sentence) > BLOCK else [sentence])
    out, held, budget = [], '', FIRST_BLOCK
    for unit in units:
        candidate = f'{held} {unit}'.strip()
        if held and len(candidate) > budget:
            out.append(held)
            held, budget = unit, min(int(budget * GROWTH), BLOCK)
        else:
            held = candidate
    if held:
        out.append(held)
    merged = []
    for piece in out:                 # a runt is not worth a synthesis call of its own
        if merged and len(piece) < MIN_BLOCK:
            merged[-1] = f'{merged[-1]} {piece}'
        else:
            merged.append(piece)
    return merged


def trim_blip(audio, rate=TTS_RATE):
    """Cut the click the model leaves in the silence after a piece, and end the piece cleanly.

    Copy of trim_blip in app/engine.py; keep the two in sync. Its measurement, over 19 readings: 11
    ended with a 10-30 ms burst, 0.15-1.2 s after the last word and sometimes as loud as the speech
    itself. In a reading made of several pieces you hear one at every join and one at the end — which
    is exactly what a reading assembled here is.
    """
    n = int(0.010 * rate)
    if len(audio) < 4 * n:
        return audio
    energy = np.sqrt((audio[:len(audio) // n * n].reshape(-1, n) ** 2).mean(axis=1))
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


def trim_file(path):
    """Apply trim_blip to a block in place, before anyone hears it."""
    try:
        with wave.open(path, 'rb') as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2 or channels != 1:
            return                       # not the 16-bit mono llama-tts writes; leave it alone
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        trimmed = trim_blip(audio, rate)
        if len(trimmed) == len(audio):
            return
        with wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(trimmed, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    except Exception as error:
        print(f'  could not trim {os.path.basename(path)}: {error}', flush=True)


def envelope(path, step=STEP):
    """How loud the block is, step by step, so the meter can follow a voice it is not recording.

    afplay reports nothing about what it is playing, so the levels are read off the file instead and
    stepped through in time with it.
    """
    try:
        with wave.open(path, 'rb') as w:
            rate = w.getframerate()
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    except Exception:
        return [0.0]
    width = max(1, int(rate * step))
    usable = len(samples) - len(samples) % width
    if usable <= 0:
        return [0.0]
    frames = samples[:usable].reshape(-1, width)
    return (np.sqrt((frames ** 2).mean(axis=1)) / 32768).tolist()


def selection(timeout=0.6):
    """Whatever is selected in the front app, by copying it and putting the clipboard back.

    Nothing can read another app's selection directly, so this does what every reader does: press
    Cmd+C and look. The board is cleared first, so "nothing was selected" is distinguishable from
    "the same text was already on the clipboard".
    """
    from AppKit import NSPasteboard, NSPasteboardTypeString
    import Quartz

    board = NSPasteboard.generalPasteboard()
    previous = board.stringForType_(NSPasteboardTypeString)
    board.clearContents()

    C_KEYCODE = 8
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, C_KEYCODE, down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.01)

    text, deadline = None, time.time() + timeout
    while time.time() < deadline:
        text = board.stringForType_(NSPasteboardTypeString)
        if text:
            break
        time.sleep(0.03)

    board.clearContents()
    if previous is not None:
        board.setString_forType_(previous, NSPasteboardTypeString)
    return text


class Reader:
    """One llama-tts kept loaded, making blocks while the previous one plays."""

    def __init__(self, voice='kreyol_f1', on_state=None, tmp=None, on_level=None):
        self.voice, self.on_state = voice, on_state or (lambda state: None)
        self.on_level = on_level or (lambda level: None)
        self.tmp = tmp or os.path.join(ROOT, 'dikte/.audio')
        self.proc = self.player = None
        self.replies = queue.Queue()
        self.ready = False
        self.error = None
        self.cancel = threading.Event()
        self.speaking = threading.Event()
        self.lock = threading.Lock()

    # loading ------------------------------------------------------------------------------------

    def preload(self):
        """Start llama-tts in the background; reading becomes available when this finishes."""
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        for path, what in ((LLAMA_TTS, 'llama-tts'), (settings.get('tts_model'), 'the voice model'),
                          (settings.get('tts_mmproj'), 'the voice projector')):
            if not os.path.exists(path):
                self.error = f'cannot find {what} at {path}; run ./setup.sh'
                print(f'  reading unavailable: {self.error}', flush=True)
                return
        os.makedirs(self.tmp, exist_ok=True)
        log = open(os.path.join(ROOT, 'dikte/llama-tts.log'), 'w')
        started = time.time()
        self.proc = subprocess.Popen([LLAMA_TTS, *tts_args(), '-p', '-', '-o', os.devnull],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                     text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        if self._reply(180) != '@@tts\tready':
            # A stock llama-tts prints no handshake. Saying so beats starting a process per block.
            self.error = ('this llama-tts has no job loop; apply llama-tts-serve.patch '
                          '(setup.sh does it) and rebuild')
            print(f'  reading unavailable: {self.error}', flush=True)
            self.shutdown()
            return
        try:
            self._make('Bonjou.', os.path.join(self.tmp, 'warmup.wav'))   # builds the embedding table
        except RuntimeError as error:
            self.error = str(error)
            print(f'  reading unavailable: {error}', flush=True)
            return
        self.ready = True
        print(f'  reading ready ({time.time() - started:.1f}s)', flush=True)

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith('@@tts'):
                self.replies.put(line.rstrip('\n'))
        self.replies.put(None)

    def _reply(self, timeout):
        try:
            return self.replies.get(timeout=timeout)
        except queue.Empty:
            return None

    def _make(self, text, out, seed=0):
        """One block. Tabs and newlines would break the line-per-job protocol, so they are gone."""
        speaker = self.voice if os.path.isabs(self.voice) else os.path.join(VOICE_DIR, f'{self.voice}.wav')
        self.proc.stdin.write(f"{out}\t{speaker}\t{seed}\t{' '.join(text.split())}\n")
        self.proc.stdin.flush()
        reply = self._reply(300)
        if reply is None:
            self.ready = False
            raise RuntimeError('the voice model stopped responding')
        fields = reply.split('\t')
        if fields[1] != 'ok':
            raise RuntimeError(fields[-1])

    # reading ------------------------------------------------------------------------------------

    def toggle(self):
        """Hold the key once to read the selection, again to stop."""
        if self.speaking.is_set():
            self.stop()
            return 'stopped'
        if not self.ready:
            return self.error or 'still loading the voice'
        text = selection()
        if not text or not text.strip():
            return 'nothing selected'
        threading.Thread(target=self._read, args=(text,), daemon=True).start()
        return None

    def _read(self, text):
        pieces = blocks(text)
        if not pieces:
            return
        self.cancel.clear()
        self.speaking.set()
        self.on_state('speaking')
        made = queue.Queue(maxsize=3)

        def produce():
            try:
                for i, piece in enumerate(pieces):
                    if self.cancel.is_set():
                        break
                    path = os.path.join(self.tmp, f'block{i % 6}.wav')
                    with self.lock:                       # one job at a time down the same pipe
                        self._make(piece, path)
                    trim_file(path)                       # before it is queued, not after
                    made.put(path)
            except Exception as error:
                print(f'  reading failed: {error}', flush=True)
            finally:
                made.put(None)

        threading.Thread(target=produce, daemon=True).start()
        try:
            while True:
                path = made.get()
                if path is None or self.cancel.is_set():
                    break
                levels = envelope(path)
                self.player = subprocess.Popen(['afplay', path], stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL)
                started = time.monotonic()
                while self.player.poll() is None:
                    step = int((time.monotonic() - started) / STEP)
                    self.on_level(levels[step] if step < len(levels) else 0.0)
                    time.sleep(STEP)
                self.player = None
                if not self.cancel.is_set():
                    self.on_level(0.0)
                    time.sleep(PAUSE)
        finally:
            self.cancel.set()                              # stop the producer if playback ended first
            self.on_level(0.0)
            self.speaking.clear()
            self.on_state('idle')

    def stop(self):
        self.cancel.set()
        player = self.player
        if player and player.poll() is None:
            player.terminate()
        self.speaking.clear()

    def shutdown(self):
        self.stop()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.ready = False
