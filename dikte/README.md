# Dikte — Kreyòl dictation, on this machine

Hold a key, speak Kreyòl, release. The text appears where your cursor is, in any app. Nothing leaves
the laptop: the audio goes to a local `whisper-server` holding the m3 model in memory, and the text goes
straight to the front app.

```bash
./start-dikte.sh                        # hold Right Command and speak
./start-dikte.sh --key right_ctrl       # if Right Command clashes with something
./start-dikte.sh --no-menu              # terminal only, no menu bar icon
python3 dikte/dikte.py --file samples/j1_16k.wav   # no mic, no permissions: check the pipe works
```

## How fast it is

Measured on an M3 Pro, model warm, through this script:

| audio | round trip |
|---|---|
| 5.0s | 0.89s |
| 5.7s | 0.90s |
| 6.9s | 0.88s |

Roughly flat regardless of length, because almost all of it is fixed cost. The server is started once
and left running; loading the 547 MB model per utterance would cost about 2 seconds every time.

## Setup

```bash
python3 -m pip install sounddevice pynput rumps pyobjc-framework-Quartz pyobjc-framework-Cocoa
```

Then grant **two** permissions to whatever you run this from — Terminal, iTerm, or your editor — in
System Settings → Privacy & Security:

- **Microphone**, to record you.
- **Accessibility**, to read the held key and to paste. Putting keystrokes into another app is exactly
  what that permission governs; there is no way around it and you should be suspicious of anything that
  claims otherwise.

Your clipboard is restored after each paste.

To check everything except the mic and those permissions, run with `--file`. It transcribes a sample
and exits.

## Why Right Command

Option is how macOS types Kreyòl accents — è, ò, à. Holding Option to dictate would fight the keyboard
you use to write the language, so the default is Right Command. `--key` takes `right_cmd`, `right_ctrl`,
`right_shift`, `right_alt`, or `f13` through `f15`.

## What the text looks like, and why

m3 was trained on one canonical form: lowercase, no punctuation, numbers spelled out. `app/asr_normalize.py`
says it plainly — casing and punctuation are a separate job for a text model on top of the transcript.

So `tidy` does only what a rule can honestly do: capitalise the opening, and close the sentence with a
full stop when it is three words or more. It does **not** place commas inside a sentence, and it does not
turn `uit è trant` into `8:30`. Restoring punctuation properly needs a Kreyòl punctuation model, which
does not exist yet. `--no-punct` turns off even the full stop.

## Sounds

Push-to-talk means you are looking at the other app, not at this one, so each state has a sound: a tick
when recording starts, a pop when text is pasted, and a lower tone when nothing was heard.

## Ports

Default 8179, its own server. `PORT=8178 ./start-dikte.sh` reuses the one the offline page starts, so
only one copy of the model is in memory when you are running both.
