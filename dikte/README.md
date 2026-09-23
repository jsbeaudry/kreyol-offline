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

The model leaves a click in the silence after each piece — 10 to 30 ms, sometimes as loud as the
speech, 0.15 to 1.2 s after the last word. In a reading made of several pieces you hear one at every
join. `trim_blip`, copied from `app/engine.py`, cuts it and fades the end; keep the two in sync.

Trimming that click also takes the trailing silence, and that silence had been the cover under which
the next block was made. After trimming, a block costs about 0.93 of its own playing time to make, so
blocks may only grow about 1.15 times each, and a 0.3 s pause sits between them — a breath between
sentences that also buys a little room. Measured on a 222-character paragraph:

| block | chars | made in | audio | extra silence |
|---|---|---|---|---|
| 0 | 53 | 5.4s | 5.6s | first sound at 5.4s |
| 1 | 52 | 5.5s | 5.8s | none |
| 2 | 62 | 6.0s | 6.5s | none |
| 3 | 52 | 5.7s | 5.6s | none |

Blocks are whole sentences. Cutting finer made a ten-character block that cost 2.7 s to produce 1.9 s
of audio; asking `split_text` for larger pieces handed back the paragraph as one 169-character block,
eleven seconds before any sound. Its sentence rule, used directly, gives neither.

The voice model takes 8.3 s to load, so it is loaded in the background when the service starts rather
than on your first selection. `--voice` picks between `kreyol_f1`, `kreyol_f2`, `kreyol_f3`,
`kreyol_m1` and `kreyol_v5`; `--no-read` skips the voice entirely and saves its 1.5 GB.

## Reglaj: which models it uses

The menu has a **Reglaj / Settings** submenu with one group per model, listing every file on this
machine that could fill it. A tick marks the one in use.

| Setting | Looks in | Kind |
|---|---|---|
| Speech to text | `models/` | `.bin` (ggml) |
| Voice | `kreyol-tts/` | `.gguf` |
| Voice projector | `kreyol-tts/` | `.gguf` named `mmproj*` |
| Who reads | `kreyol-tts/voices/` | `.wav` |

Drop another quantisation beside the current one — `ggml-oswald-m3-q8_0.bin`, say, or a different
`-Q5_K_M.gguf` — and it appears in the menu next time it opens. Nothing is copied or converted here;
`convert/` does that and `setup.sh` fetches the published ones. **Open model folders** in the same
submenu reveals both in the Finder.

Changing the speech model or the voice means loading it again, so the service restarts (a couple of
seconds for speech, about eight for the voice). Changing who reads is only another reference clip, so
it applies to the next reading with no restart.

Choices live in `dikte/settings.json`, written the way `app/settings.py` writes the page's: to a
temporary file and then renamed, so there is never a half-written one. Anything unknown, of the wrong
kind, or not actually on disk is dropped on the way in — the cost of a bad value here is not an
exception but a model server that will not start, with the reason buried in a log. A model deleted
behind the app's back is reported at startup and the default used instead.

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
