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
import subprocess
import sys
import threading
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_TTS = os.path.join(ROOT, 'llama.cpp/build/bin/llama-tts')
TTS_MODEL = os.path.join(ROOT, 'kreyol-tts/qwen3-tts-1.7b-kreyol-Q4_K_M.gguf')
TTS_MMPROJ = os.path.join(ROOT, 'kreyol-tts/mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf')
VOICE_DIR = os.path.join(ROOT, 'kreyol-tts/voices')
# -c 2048 for the same reason engine.py gives: otherwise a 32k KV cache costs 3.5 GB for a few hundred
# tokens. The sampling settings match the page so a sentence sounds the same in both.
TTS_ARGS = ['-m', TTS_MODEL, '--mmproj', TTS_MMPROJ, '-c', '2048', '-ngl', '99',
            '--temp', '0.9', '--top-k', '50', '--top-p', '1.0', '--repeat-penalty', '1.05']

FIRST_BLOCK = 45         # small, so the sound starts sooner
BLOCK = 220              # what kreyol_text.split_text uses by default
MAX_CHARS = 5000         # a whole document selected by accident should not become a ten-minute reading
# Making a block costs about 0.73 of the time it takes to play, so while one block plays there is room
# to make the next one up to about 1.37 times its length. Grow slower than that and the sound never
# catches up with the making; jump straight to full-size blocks and there is a gap after the first.
GROWTH = 1.35
STEP = 0.08              # how often the level is reported while a block plays

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
    units = kreyol_text.split_text(text, max_chars=FIRST_BLOCK)   # the finest split worth reading
    out, held, budget = [], '', FIRST_BLOCK
    for unit in units:
        if held and len(held) + 1 + len(unit) > budget:
            out.append(held)
            held, budget = unit, min(int(budget * GROWTH), BLOCK)
        else:
            held = f'{held} {unit}'.strip()
    if held:
        out.append(held)
    return out


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
        for path, what in ((LLAMA_TTS, 'llama-tts'), (TTS_MODEL, 'the voice model')):
            if not os.path.exists(path):
                self.error = f'cannot find {what} at {path}; run ./setup.sh'
                print(f'  reading unavailable: {self.error}', flush=True)
                return
        os.makedirs(self.tmp, exist_ok=True)
        log = open(os.path.join(ROOT, 'dikte/llama-tts.log'), 'w')
        started = time.time()
        self.proc = subprocess.Popen([LLAMA_TTS, *TTS_ARGS, '-p', '-', '-o', os.devnull],
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
        speaker = os.path.join(VOICE_DIR, f'{self.voice}.wav')
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
