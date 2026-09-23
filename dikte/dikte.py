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
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

import settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_SERVER = os.path.join(ROOT, 'whisper.cpp/build/bin/whisper-server')
RATE = 16000                       # what whisper wants; resampling later would only add latency
MIN_SECONDS = 0.35                 # shorter than this is a stray key tap, not speech
MAX_SECONDS = 120
# Left Command is Cmd+C, Cmd+V, Cmd+Tab all day long, so reading fires only when the key was held
# alone, with nothing else pressed, for at least this long. A shortcut fails both tests.
HOLD_SECONDS = 0.5

# Modifier keys that are safe to hold on a Kreyòl keyboard. Option is deliberately absent from the
# defaults: it is the accent key.
KEY_NAMES = {'right_cmd': 'cmd_r', 'right_ctrl': 'ctrl_r', 'right_shift': 'shift_r',
             'right_alt': 'alt_r', 'left_cmd': 'cmd_l', 'left_ctrl': 'ctrl_l',
             'f13': 'f13', 'f14': 'f14', 'f15': 'f15'}

# The page answers this with its Grille: dots lighting from the centre out with the voice, red going
# in, ink coming out. No sound anywhere — a beep over your own dictation is noise, and a beep in a
# meeting is worse. The menu bar has one character to work with, so the level goes there.
METER = ' ▁▂▃▄▅▆▇█'
FLOOR = 0.004              # below this is a quiet room, not speech


def meter(level):
    """One character standing for how loud it is now, on the same scale the page paints."""
    if level <= FLOOR:
        return METER[0]
    # Loudness is logarithmic; a linear bar would sit near the bottom for all ordinary speech.
    loud = max(0.0, min(1.0, (math.log10(level) - math.log10(FLOOR)) / (math.log10(0.5) - math.log10(FLOOR))))
    return METER[min(len(METER) - 1, 1 + int(loud * (len(METER) - 2)))]


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
    model = settings.get('stt_model')
    for path, what in ((WHISPER_SERVER, 'whisper-server'), (model, 'the speech model')):
        if not os.path.exists(path):
            sys.exit(f'cannot find {what} at {path}\nRun ./setup.sh first.')
    log = open(os.path.join(ROOT, 'dikte/whisper-server.log'), 'w')
    process = subprocess.Popen([WHISPER_SERVER, '-m', model, '-l', 'ht', '--host', '127.0.0.1',
                                '--port', str(port)], stdout=log, stderr=subprocess.STDOUT)
    print(f'loading {settings.label(model)} on port {port}', end='', flush=True)
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


class Holds:
    """Decide what a modifier key press meant: a deliberate hold, or part of a shortcut.

    Left Command is pressed all day for Cmd+C, Cmd+V and Cmd+Tab. The only safe reading of "hold Left
    Command" is: it went down on its own, nothing else was pressed while it was down, and it stayed
    down a while. A shortcut fails the second test, a tap fails the third.
    """

    def __init__(self, keys, clock=time.monotonic):
        self.keys = {key for key in keys if key is not None}
        self.clock = clock
        self.key, self.since, self.alone = None, 0.0, True

    def press(self, key):
        """True when one of the watched keys has just gone down by itself."""
        if self.key is None:
            if key in self.keys:
                self.key, self.since, self.alone = key, self.clock(), True
                return True
            return False
        if key != self.key:
            self.alone = False          # something else joined it: this is a shortcut being typed
        return False

    def release(self, key):
        """(key that was held, whether nothing else was pressed, how long it was down)."""
        if key != self.key:
            return None, False, 0.0
        held, alone, seconds = self.key, self.alone, self.clock() - self.since
        self.key = None
        return held, alone, seconds


class Dictation:
    """Hold the key: record. Release it: transcribe, tidy, paste."""

    def __init__(self, port, punctuate=True, on_state=None, on_level=None):
        self.port, self.punctuate, self.on_state = port, punctuate, on_state or (lambda s: None)
        self.on_level = on_level or (lambda level: None)
        self.frames, self.stream, self.lock = [], None, threading.Lock()
        self.recording = False
        self.last_level = 0.0

    def _collect(self, data, frames, timing, status):
        with self.lock:
            if not self.recording:
                return
            chunk = bytes(data)
            self.frames.append(chunk)
        # The callback runs about thirty times a second; the menu bar does not need that.
        now = time.monotonic()
        if now - self.last_level < 0.08:
            return
        self.last_level = now
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if samples.size:
            self.on_level(float(np.sqrt((samples ** 2).mean())) / 32768)

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
            self.on_state('idle')
            return
        self.on_state('recording')

    def cancel(self):
        """Throw the recording away: the key turned out to be part of a shortcut, not dictation."""
        if not self.recording:
            return
        with self.lock:
            self.recording = False
            self.frames = []
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.on_state('idle')

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
            self.on_state('idle')
            return
        elapsed = time.time() - started
        if not text:
            print(f'  ({seconds:.1f}s, nothing heard)')
        else:
            print(f'  [{seconds:.1f}s → {elapsed:.2f}s] {text}')
            paste(text)
        self.on_state('idle')




class Service:
    """The two things that can be running: the model server and the key listener.

    They are kept together so the menu can stop both — freeing the 547 MB the model holds — and start
    them again later without quitting the app.
    """

    def __init__(self, port, key, punctuate=True, on_state=None, read_key=None, voice='kreyol_f1'):
        self.port, self.key, self.punctuate = port, key, punctuate
        self.read_key, self.voice = read_key, voice
        self.on_state = on_state or (lambda state: None)
        self.on_level = lambda level: None
        self.server = self.listener = self.dictation = self.reader = None

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
        self.dictation = Dictation(self.port, self.punctuate, lambda state: self.on_state(state),
                                   on_level=lambda level: self.on_level(level))
        dictate_key = getattr(keyboard.Key, KEY_NAMES[self.key])
        read_key = getattr(keyboard.Key, KEY_NAMES[self.read_key]) if self.read_key else None

        if read_key is not None:
            from reader import Reader
            self.reader = Reader(self.voice, on_state=lambda state: self.on_state(state),
                                 on_level=lambda level: self.on_level(level))
            self.reader.preload()          # 8.3 s, in the background, so the first selection is not slow

        def guard(action):
            """An exception here would stop the listener and print nothing, leaving the key dead."""
            try:
                action()
            except Exception as error:
                print(f'  {type(error).__name__}: {error}', flush=True)

        holds = Holds([dictate_key, read_key])

        def on_press(key):
            # Dictation starts the moment the key goes down: push-to-talk that waited would clip you.
            if holds.press(key) and key == dictate_key:
                guard(self.dictation.start)

        def on_release(key):
            held, alone, seconds = holds.release(key)
            if held is None:
                return
            if held == dictate_key:
                # Right Command with something else was a shortcut; throw the recording away.
                guard(self.dictation.stop if alone else self.dictation.cancel)
            elif alone and seconds >= HOLD_SECONDS:
                guard(self.read_selection)

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        if not self.listener.running:
            self.stop()
            raise RuntimeError('the key listener would not start; try python3 dikte/diagnose.py')
        self.on_state('idle')

    def restart(self):
        """Stop and start again, for a setting that means loading a model afresh."""
        self.stop()
        self.start()

    def read_selection(self):
        note = self.reader.toggle() if self.reader else 'reading is switched off'
        if note:
            print(f'  {note}', flush=True)

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.dictation = None
        if self.reader:
            self.reader.shutdown()         # gives back the 1.5 GB the voice model holds
            self.reader = None
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
    if service.read_key:
        print(f'select text and hold {service.read_key.replace("_", " ")} alone to hear it read.')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print('\nstopping')
    finally:
        service.stop()


def settings_menu(service, on_change):
    """A submenu per setting, listing the files on this machine that could fill it.

    A tick marks the one in use. Choosing a different speech or voice model means loading it again, so
    the service restarts; choosing a different speaker is only another reference clip, and applies to
    the next reading.
    """
    import rumps

    root = rumps.MenuItem('Reglaj / Settings')
    marks = []                                     # (setting, path, item) to tick and untick together

    def mark():
        for name, path, item in marks:
            item.state = 1 if settings.get(name) == path else 0

    def apply(name, path):
        def handler(_):
            settings.update({name: path})
            mark()
            try:
                if name in settings.NEEDS_RESTART and service.running:
                    service.restart()              # that model has to be loaded again
                elif name == 'speaker' and service.reader:
                    service.reader.voice = path    # no reload; the next reading uses it
            except Exception as error:
                service.stop()
                rumps.alert('Dikte', f'{type(error).__name__}: {error}')
            on_change()
        return handler

    busy = {'name': None}                          # one download at a time

    def title_for(option):
        note = f' — {option["note"]}' if option['note'] else ''
        if option['local']:
            return f'{option["label"]}{note}'
        return f'{option["label"]} — {option["mb"]:,} MB, telechaje{note}'

    def download(name, option, item):
        """Fetch a published model, then switch to it. Gigabytes and minutes, so it asks first."""
        def handler(_):
            if busy['name']:
                rumps.alert('Dikte', 'Gen yon telechajman k ap fèt deja. / A download is already running.')
                return
            if rumps.alert('Dikte', f'Telechaje {option["label"]}?\n\n{option["mb"]:,} MB '
                                    f'soti nan {settings.REPOS[name]}.',
                           ok='Telechaje / Download', cancel='Anile / Cancel') != 1:
                return
            busy['name'] = name
            plain = item.title

            def run():
                try:
                    settings.fetch(name, option['path'],
                                   progress=lambda f: setattr(item, 'title',
                                                              f'{option["label"]} — {f * 100:.0f}%'))
                    option['local'] = True
                    item.title = title_for(option)
                    item.set_callback(apply(name, option['path']))
                    marks.append((name, option['path'], item))
                    apply(name, option['path'])(None)     # having fetched it, use it
                except Exception as error:
                    item.title = plain
                    rumps.alert('Dikte', f'{type(error).__name__}: {error}')
                finally:
                    busy['name'] = None
            threading.Thread(target=run, daemon=True).start()
        return handler

    for name, (_, title, folder, suffix) in settings.FIELDS.items():
        group = rumps.MenuItem(title)
        found = settings.options(name)
        for option in found:
            item = rumps.MenuItem(title_for(option))
            item.set_callback(apply(name, option['path']) if option['local']
                              else download(name, option, item))
            if option['local']:
                marks.append((name, option['path'], item))
            group.add(item)
        if not found:
            empty = rumps.MenuItem(f'pa gen {suffix} nan {os.path.basename(folder)}/')
            empty.set_callback(None)
            group.add(empty)
        root.add(group)

    def open_folders(_):
        for folder in ('models', 'kreyol-tts'):
            subprocess.Popen(['open', os.path.join(ROOT, folder)])

    def reset(_):
        settings.reset()
        mark()
        try:
            if service.running:
                service.restart()
        except Exception as error:
            service.stop()
            rumps.alert('Dikte', f'{type(error).__name__}: {error}')
        on_change()

    root.add(rumps.separator)
    root.add(rumps.MenuItem('Louvri dosye modèl yo / Open model folders', callback=open_folders))
    root.add(rumps.MenuItem('Remete / Reset to defaults', callback=reset))
    mark()
    return root


def run_menu(service, key_name):
    """A menu bar icon that says what the service is doing, and one item that starts or stops it."""
    import rumps

    ICONS = {'off': '⏸', 'starting': '⋯', 'idle': '🎙', 'recording': '🔴', 'working': '⋯',
             'speaking': '🔊'}
    app = rumps.App('Dikte', title=ICONS['off'], quit_button=None)
    toggle = rumps.MenuItem('Kòmanse / Start')
    spoken = key_name.replace('_', ' ')
    read_name = (service.read_key or '').replace('_', ' ')
    hint = rumps.MenuItem(f'Kenbe {spoken} epi pale')
    read_hint = rumps.MenuItem(f'Chwazi yon tèks, kenbe {read_name} pou tande l')

    current = {'state': 'off'}

    def refresh(state):
        current['state'] = state
        app.title = ICONS.get(state, ICONS['idle'])
        toggle.title = 'Kanpe / Stop' if service.running else 'Kòmanse / Start'
        hint.title = (f'Kenbe {spoken} epi pale — {settings.label(settings.get("stt_model"))}'
                      if service.running else 'Sèvis la kanpe / Service stopped')
        if service.reader is None:
            read_hint.title = 'Lekti pa disponib / Reading off'
        elif service.reader.error:
            read_hint.title = f'Lekti: {service.reader.error[:44]}'
        elif not service.reader.ready:
            read_hint.title = 'Vwa a ap chaje / Voice loading…'
        else:
            read_hint.title = (f'Chwazi yon tèks, kenbe {read_name} pou tande l — '
                               f'{settings.label(settings.get("speaker"))}')

    service.on_state = refresh

    def level(value):
        if current['state'] in ('recording', 'speaking'):
            app.title = meter(value)

    service.on_level = level

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
    hint.set_callback(None)                        # labels, not buttons
    read_hint.set_callback(None)
    app.menu = [toggle, hint, read_hint, None,
                settings_menu(service, lambda: refresh(current['state'])), None,
                rumps.MenuItem('Kite / Quit', callback=quit_app)]
    # The voice finishes loading seconds after the service starts, so the label has to catch up. Repeat
    # the state rather than assuming one, or this would clear the icon in the middle of a recording.
    rumps.Timer(lambda _: refresh(current['state']), 2).start()

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
    p.add_argument('--read-key', default='left_cmd', choices=sorted(KEY_NAMES),
                   help='hold this alone to read the selected text aloud (default: left_cmd)')
    p.add_argument('--voice', default=None,
                   help='kreyol_f1, kreyol_f2, kreyol_f3, kreyol_m1 or kreyol_v5; remembered after')
    p.add_argument('--no-read', action='store_true', help='dictation only; do not load the voice')
    p.add_argument('--no-punct', action='store_true', help='do not add a full stop')
    p.add_argument('--no-menu', action='store_true', help='no menu bar icon, just the terminal')
    p.add_argument('--file', help='transcribe this wav and exit, instead of listening to the mic')
    args = p.parse_args()
    if args.read_key == args.key:
        p.error('--read-key and --key must be different')
    if args.voice:
        # The flag sets the stored setting, so the menu and the next run agree with it.
        chosen = os.path.join(ROOT, 'kreyol-tts/voices', f'{args.voice}.wav')
        if not os.path.exists(chosen):
            p.error(f'no such voice: {args.voice}')
        settings.update({'speaker': chosen})
    gone = settings.missing()
    if gone:
        print('these settings point at files that are not there, using the defaults instead:',
              flush=True)
        for name, path in gone.items():
            print(f'  {name}: {path}', flush=True)
        settings.update({name: settings.FIELDS[name][0] for name in gone})

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

    service = Service(args.port, args.key, punctuate=not args.no_punct,
                      read_key=None if args.no_read else args.read_key,
                      voice=settings.get('speaker'))
    try:
        (run_headless if args.no_menu else run_menu)(service, args.key)
    finally:
        service.stop()


if __name__ == '__main__':
    main()
