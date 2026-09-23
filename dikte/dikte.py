"""Dikte: hold a key, speak Kreyòl, and the text lands where your cursor is.

Dictation for Haitian Creole that never leaves this machine. The audio goes to a whisper-server holding
the m3 model in memory, and the text goes straight into whatever app you are typing in.

Measured on an M3 Pro with the model warm: a 5 to 7 second utterance comes back in 0.9 seconds, near
enough the same whatever its length. That is what makes this feel like dictation rather than a
transcription job, and it is why the server is started once and left running rather than per utterance.

Three things worth knowing:

  The key is Right Command, not Right Option. Option is how macOS types Kreyòl accents (è, ò, à), so
  holding Option to dictate would fight the keyboard you use to write the language.

  The transcript arrives lowercase, unpunctuated, with numbers spelled out, because that is the one form
  m3 was trained on (see app/asr_normalize.py, which says so outright: casing and punctuation are a
  separate job for a text model). `tidy` opens with a capital and closes the sentence. It cannot put
  commas inside one, and "uit è trant" will not become "8:30".

  Pasting needs Accessibility permission, because putting keystrokes into another app is exactly what
  that permission governs. Your clipboard is put back afterwards.

    python3 dikte/dikte.py                      # hold Right Command, speak, release
    python3 dikte/dikte.py --key right_ctrl     # if Right Command clashes with something
    python3 dikte/dikte.py --file samples/j1_16k.wav   # no mic, no permissions: check the pipe works
"""
import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_SERVER = os.path.join(ROOT, 'whisper.cpp/build/bin/whisper-server')
MODEL = os.path.join(ROOT, 'models/ggml-oswald-m3-q5_0.bin')
RATE = 16000                       # what whisper wants; resampling later would only add latency
MIN_SECONDS = 0.35                 # shorter than this is a stray key tap, not speech
MAX_SECONDS = 120

# Modifier keys that are safe to hold on a Kreyòl keyboard. Option is deliberately absent from the
# defaults: it is the accent key.
KEY_NAMES = {'right_cmd': 'cmd_r', 'right_ctrl': 'ctrl_r', 'right_shift': 'shift_r',
             'right_alt': 'alt_r', 'f13': 'f13', 'f14': 'f14', 'f15': 'f15'}

SOUNDS = {'start': '/System/Library/Sounds/Tink.aiff',
          'done': '/System/Library/Sounds/Pop.aiff',
          'empty': '/System/Library/Sounds/Funk.aiff'}


def cue(name):
    """A sound, because in push-to-talk you are looking at the other app, not at this one."""
    path = SOUNDS.get(name)
    if path and os.path.exists(path):
        subprocess.Popen(['afplay', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wav_bytes(frames, rate=RATE):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames)
    return buffer.getvalue()


def whisper_text(data, port, timeout=300):
    """Same multipart shape app/engine.py uses, so both talk to the server identically."""
    boundary = uuid.uuid4().hex
    body = b''.join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0.0\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'.encode(),
        data,
        f'\r\n--{boundary}--\r\n'.encode(),
    ])
    request = urllib.request.Request(f'http://127.0.0.1:{port}/inference', data=body,
                                     headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return ' '.join(json.loads(response.read()).get('text', '').split())


def server_is_up(port):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2)
        return True
    except urllib.error.HTTPError:
        return True                # answering at all is enough; the root path has no handler
    except Exception:
        return False


def start_server(port, quiet=True):
    """Start whisper-server if nothing is answering, and wait until it is."""
    if server_is_up(port):
        print(f'using the whisper-server already on port {port}')
        return None
    for path, what in ((WHISPER_SERVER, 'whisper-server'), (MODEL, 'the m3 model')):
        if not os.path.exists(path):
            sys.exit(f'cannot find {what} at {path}\nRun ./setup.sh first.')
    log = open(os.path.join(ROOT, 'dikte/whisper-server.log'), 'w')
    process = subprocess.Popen([WHISPER_SERVER, '-m', MODEL, '-l', 'ht', '--host', '127.0.0.1',
                                '--port', str(port)], stdout=log, stderr=subprocess.STDOUT)
    print(f'loading m3 on port {port}', end='', flush=True)
    for _ in range(180):
        if process.poll() is not None:
            sys.exit(f'\nwhisper-server stopped; see dikte/whisper-server.log')
        if server_is_up(port):
            print(' — ready')
            return process
        print('.', end='', flush=True)
        time.sleep(1)
    process.terminate()
    sys.exit('\nwhisper-server did not start within 3 minutes')


def tidy(text, punctuate=True):
    """Open with a capital, close with a full stop. Everything else needs a model, not a rule.

    m3 emits one canonical form: lowercase, no punctuation, digits spelled out. Sentence-internal commas
    and real number formatting would take a punctuation-restoration model, which does not exist for
    Kreyòl yet, so this does not pretend to supply them.
    """
    text = ' '.join(text.split())
    if not text:
        return ''
    text = text[0].upper() + text[1:]
    # A fragment dictated into a search box does not want a full stop; a sentence does.
    if punctuate and len(text.split()) >= 3 and text[-1] not in '.!?:,;':
        text += '.'
    return text


def paste(text):
    """Put text at the cursor, then give the clipboard back to whatever was on it."""
    from AppKit import NSPasteboard, NSPasteboardTypeString
    import Quartz

    board = NSPasteboard.generalPasteboard()
    previous = board.stringForType_(NSPasteboardTypeString)
    board.clearContents()
    board.setString_forType_(text, NSPasteboardTypeString)

    V_KEYCODE = 9
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.01)

    def restore():
        time.sleep(0.4)            # the paste has to land before the clipboard changes under it
        board.clearContents()
        if previous is not None:
            board.setString_forType_(previous, NSPasteboardTypeString)
    threading.Thread(target=restore, daemon=True).start()


def accessibility_ok():
    from ApplicationServices import AXIsProcessTrusted
    return bool(AXIsProcessTrusted())


class Dictation:
    """Hold the key: record. Release it: transcribe, tidy, paste."""

    def __init__(self, port, punctuate=True, on_state=None):
        self.port, self.punctuate, self.on_state = port, punctuate, on_state or (lambda s: None)
        self.frames, self.stream, self.lock = [], None, threading.Lock()
        self.recording = False

    def _collect(self, data, frames, timing, status):
        with self.lock:
            if self.recording:
                self.frames.append(bytes(data))

    def start(self):
        import sounddevice as sd
        if self.recording:
            return                 # holding a key repeats on_press; only the first one counts
        with self.lock:
            self.frames, self.recording = [], True
        try:
            self.stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype='int16',
                                            callback=self._collect)
            self.stream.start()
        except Exception as error:
            # Losing this exception into the listener thread is how "holding the key does nothing"
            # happens with no explanation. A denied microphone is the usual cause.
            self.recording = False
            print(f'  cannot open the microphone: {type(error).__name__}: {error}\n'
                  f'  System Settings → Privacy & Security → Microphone, add the app running this,\n'
                  f'  then quit it completely and reopen it.', flush=True)
            cue('empty')
            self.on_state('idle')
            return
        self.on_state('recording')
        cue('start')

    def stop(self):
        if not self.recording:
            return
        with self.lock:
            self.recording = False
            data = b''.join(self.frames)
            self.frames = []
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        seconds = len(data) / 2 / RATE
        if seconds < MIN_SECONDS:
            self.on_state('idle')
            return
        if seconds > MAX_SECONDS:
            data = data[:int(MAX_SECONDS * RATE * 2)]
        threading.Thread(target=self._finish, args=(data, seconds), daemon=True).start()

    def _finish(self, data, seconds):
        self.on_state('working')
        started = time.time()
        try:
            # The microphone gives raw PCM; the server decodes files, so it needs the RIFF header.
            text = tidy(whisper_text(wav_bytes(data), self.port), self.punctuate)
        except Exception as error:
            failed = os.path.join(ROOT, 'dikte/failed.wav')
            try:
                with open(failed, 'wb') as f:
                    f.write(wav_bytes(data))
            except Exception:
                failed = None
            print(f'  transcription failed: {type(error).__name__}: {error}'
                  + (f'\n  the audio was kept at {failed} — play it to hear what was captured'
                     if failed else ''), flush=True)
            cue('empty')
            self.on_state('idle')
            return
        elapsed = time.time() - started
        if not text:
            print(f'  ({seconds:.1f}s, nothing heard)')
            cue('empty')
        else:
            print(f'  [{seconds:.1f}s → {elapsed:.2f}s] {text}')
            paste(text)
            cue('done')
        self.on_state('idle')




class Service:
    """The two things that can be running: the model server and the key listener.

    They are kept together so the menu can stop both — freeing the 547 MB the model holds — and start
    them again later without quitting the app.
    """

    def __init__(self, port, key, punctuate=True, on_state=None):
        self.port, self.key, self.punctuate = port, key, punctuate
        self.on_state = on_state or (lambda state: None)
        self.server = self.listener = self.dictation = None

    @property
    def running(self):
        return self.listener is not None and self.listener.running

    def start(self):
        if self.running:
            return
        self.on_state('starting')
        self.server = start_server(self.port)      # None when reusing a server already listening
        from pynput import keyboard
        # The lambda defers to the current on_state, so the menu can replace it after construction.
        self.dictation = Dictation(self.port, self.punctuate, lambda state: self.on_state(state))
        hotkey = getattr(keyboard.Key, KEY_NAMES[self.key])

        def on(action):
            """An exception here would stop the listener and print nothing, leaving the key dead."""
            def handler(key):
                if key != hotkey:
                    return
                try:
                    action()
                except Exception as error:
                    print(f'  {type(error).__name__}: {error}', flush=True)
            return handler

        self.listener = keyboard.Listener(on_press=on(self.dictation.start),
                                          on_release=on(self.dictation.stop))
        self.listener.start()
        if not self.listener.running:
            self.stop()
            raise RuntimeError('the key listener would not start; try python3 dikte/diagnose.py')
        self.on_state('idle')

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.dictation = None
        if self.server:
            # Only a server this process started. One that was already listening belongs to something
            # else — the offline page, most likely — and is not ours to kill.
            self.server.terminate()
            try:
                self.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server.kill()
            self.server = None
        self.on_state('off')


def run_headless(service, key_name):
    service.start()
    print(f'hold {key_name.replace("_", " ")} and speak. Ctrl+C to stop.')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print('\nstopping')
    finally:
        service.stop()


def run_menu(service, key_name):
    """A menu bar icon that says what the service is doing, and one item that starts or stops it."""
    import rumps

    ICONS = {'off': '⏸', 'starting': '⋯', 'idle': '🎙', 'recording': '🔴', 'working': '⋯'}
    app = rumps.App('Dikte', title=ICONS['off'], quit_button=None)
    toggle = rumps.MenuItem('Kòmanse / Start')
    hint = rumps.MenuItem(f'Kenbe {key_name.replace("_", " ")} epi pale')

    def refresh(state):
        app.title = ICONS.get(state, ICONS['idle'])
        toggle.title = 'Kanpe / Stop' if service.running else 'Kòmanse / Start'
        hint.title = (f'Kenbe {key_name.replace("_", " ")} epi pale' if service.running
                      else 'Sèvis la kanpe / Service stopped')

    service.on_state = refresh

    def toggled(_):
        try:
            service.stop() if service.running else service.start()
        except Exception as error:
            service.stop()
            rumps.alert('Dikte', f'{type(error).__name__}: {error}')
        refresh('idle' if service.running else 'off')

    def quit_app(_):
        service.stop()
        rumps.quit_application()

    toggle.set_callback(toggled)
    hint.set_callback(None)                        # a label, not a button
    app.menu = [toggle, hint, None, rumps.MenuItem('Kite / Quit', callback=quit_app)]

    # Start on launch: a dictation app that opens switched off is one you forget to switch on.
    toggled(None)
    app.run()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--key', default='right_cmd', choices=sorted(KEY_NAMES),
                   help='hold this to dictate (default: right_cmd; Option is the Kreyòl accent key)')
    p.add_argument('--port', type=int, default=8179,
                   help='whisper-server port; 8178 shares the one the offline page starts')
    p.add_argument('--no-punct', action='store_true', help='do not add a full stop')
    p.add_argument('--no-menu', action='store_true', help='no menu bar icon, just the terminal')
    p.add_argument('--file', help='transcribe this wav and exit, instead of listening to the mic')
    args = p.parse_args()

    if args.file:
        server = start_server(args.port)
        try:
            with open(args.file, 'rb') as f:
                data = f.read()
            started = time.time()
            print(tidy(whisper_text(data, args.port), not args.no_punct))
            print(f'({time.time() - started:.2f}s)')
        finally:
            if server:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
        return

    if not accessibility_ok():
        message = ('Dikte needs Accessibility permission to read the hotkey and to paste.\n\n'
                   'System Settings → Privacy & Security → Accessibility → add the app running this '
                   '(Dikte, or your terminal if you are running the script), switch it on, then quit '
                   'it completely and open it again.')
        if not args.no_menu:
            # From the Finder there is no terminal and no Dock icon, so a message printed to the log
            # is a silent failure. Put it on screen.
            try:
                import rumps
                rumps.alert('Dikte', message)
            except Exception:
                pass
        sys.exit('\n' + message + '\n\nTo check the rest works without granting anything:\n'
                 '  python3 dikte/dikte.py --file samples/j1_16k.wav')

    service = Service(args.port, args.key, punctuate=not args.no_punct)
    try:
        (run_headless if args.no_menu else run_menu)(service, args.key)
    finally:
        service.stop()


if __name__ == '__main__':
    main()
