# Dikte — Kreyòl dictation and reading, on this machine

Two keys, opposite directions, in any app. Nothing leaves the laptop.

| | |
|---|---|
| **Hold Right Command**, speak, release | Your Kreyòl appears where the cursor is |
| **Select text, hold Left Command** | It is read aloud in a Kreyòl voice |

The speech goes to a local `whisper-server` holding the m3 model in memory; the reading goes to a local
`llama-tts` holding the Kreyòl voice model.

## Reading the selection

Hold Left Command on its own for half a second and whatever is selected is read aloud. Hold it again to
stop. Nothing can read another app's selection directly, so Dikte copies it (Cmd+C) and puts your
clipboard back.

**Left Command is also Cmd+C, Cmd+V and Cmd+Tab**, which is the whole difficulty. Reading fires only
when the key went down by itself, nothing else was pressed while it was down, and it stayed down at
least half a second. A shortcut fails the second test; a tap fails the third. `dikte/test_dikte.py`
checks exactly that, because the failure mode is not an error message — it is a voice reading your
clipboard every time you copy something.

Making speech runs at about 1.1 to 1.3 times real time: fast enough to stay ahead of playback, not fast
enough to make a paragraph before it starts. So the selection is cut into blocks that start small and
grow, each at most about 1.35 times the last, which is all that can be made while the previous one
plays. Measured on a 222-character paragraph: the first words at **3.7 s**, then no silence at all.

| block | chars | made in | audio | silence before it |
|---|---|---|---|---|
| 0 | 42 | 3.7s | 5.2s | first sound at 3.7s |
| 1 | 55 | 4.8s | 6.1s | none |
| 2 | 70 | 6.4s | 9.0s | none |
| 3 | 52 | 5.2s | 6.6s | none |

Going straight to full-size blocks instead puts a 2.2-second gap after the first one.

The voice model takes 8.3 s to load, so it is loaded in the background when the service starts rather
than on your first selection. `--voice` picks between `kreyol_f1`, `kreyol_f2`, `kreyol_f3`,
`kreyol_m1` and `kreyol_v5`; `--no-read` skips the voice entirely and saves its 1.5 GB.

## Dictation

```bash
./build-dikte-app.sh                    # build Dikte.app, then open it from the Finder
./start-dikte.sh                        # or run it from a terminal instead
./start-dikte.sh --key right_ctrl       # if Right Command clashes with something
./start-dikte.sh --no-menu              # terminal only, no menu bar icon
python3 dikte/dikte.py --file samples/j1_16k.wav   # no mic, no permissions: check the pipe works
```

## The app

`./build-dikte-app.sh` makes `~/Applications/Dikte.app`: a menu bar icon and nothing else — no Dock
icon, no window, no terminal left open. The icon is the state.

| | |
|---|---|
| 🎙 | listening for the keys |
| 🔴 | recording you |
| 🔊 | reading the selection |
| ⋯ | starting, or working |
| ⏸ | service stopped |

Its one menu item starts and stops the service. Stopping shuts down both models too — the 547 MB
speech model and the 1.5 GB voice — and starting reloads them. Quit from the same menu.

The app is a thin launcher around `dikte/dikte.py` in this folder, not a copy, so editing the script
changes the app without rebuilding. The interpreter and folder are written in as absolute paths,
because an app launched from the Finder gets a minimal `PATH`, none of your shell's setup, and no
pyenv shims — `python3` there is not the `python3` you have been using.

**Do not run the app and the terminal script at the same time.** Both install a key listener, both
would record, and you would get your words pasted twice.

Permissions belong to whichever one you use. Granting Accessibility to Dikte.app means you can take it
away from Terminal, which is the narrower arrangement: a dictation app that can type into other apps,
rather than a terminal that can.

Its log is `~/Library/Logs/Dikte.log`. That is where a failure to start goes, since there is no
terminal to print to.

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

Then grant **two** permissions to whatever you run it from — Dikte.app, or your terminal if you use the
script — in System Settings → Privacy & Security:

- **Microphone**, to record you.
- **Accessibility**, to read the held key and to paste. Putting keystrokes into another app is exactly
  what that permission governs; there is no way around it and you should be suspicious of anything that
  claims otherwise.

Your clipboard is restored after each paste.

To check everything except the mic and those permissions, run with `--file`. It transcribes a sample
and exits.

## Why Right Command

Option is how macOS types Kreyòl accents — è, ò, à. Holding Option to dictate would fight the keyboard
you use to write the language, so the default is Right Command, with Left Command for reading.

`--key` and `--read-key` both take `right_cmd`, `right_ctrl`, `right_shift`, `right_alt`, `left_cmd`,
`left_ctrl`, or `f13` through `f15`, and must differ. The right-hand keys are the safer ones to hold:
dictation starts the instant the key goes down, so binding it to a key you use in shortcuts would
record every time you pressed it. Reading waits for a deliberate hold, which is why Left Command is
safe for it and would not be safe for dictation.

## What the text looks like, and why

m3 was trained on one canonical form: lowercase, no punctuation, numbers spelled out. `app/asr_normalize.py`
says it plainly — casing and punctuation are a separate job for a text model on top of the transcript.

So `tidy` does only what a rule can honestly do: capitalise the opening, and close the sentence with a
full stop when it is three words or more. It does **not** place commas inside a sentence, and it does not
turn `uit è trant` into `8:30`. Restoring punctuation properly needs a Kreyòl punctuation model, which
does not exist yet. `--no-punct` turns off even the full stop.

## The meter, not beeps

An earlier version played system sounds at each step. That was wrong for the same reason the page has
none: a beep over your own dictation is noise, and a beep in a meeting is worse. The page answers this
with its **Grille** — dots lighting from the centre out with the voice, red going in, ink coming out.

Dikte does the same thing with the one character the menu bar gives it. While you speak, the icon
becomes a live level meter reading off the microphone; while it reads to you, the same meter follows the
voice it is playing. `afplay` reports nothing about what it plays, so those levels are read off the
block's own audio and stepped through in time with it.

```
         ▄▄▆▆▆▃▅▅ ▅▅ ▄▅▅▅▆▆▃ ▄▅▆▄▃▅▅▄▂▃▅▂▁▅▅▅▅▃ ▄▅▄▃▄▄▅▅▄▄▅▄▁
```

That is a real 5.7-second Kreyòl clip: the silence at each end is the silence in the file. The scale is
logarithmic, because a linear one leaves ordinary speech sitting near the bottom.

## Ports

Default 8179, its own server. `PORT=8178 ./start-dikte.sh` reuses the one the offline page starts, so
only one copy of the model is in memory when you are running both.
