"""Find out why dictation is not starting: what the key listener sees, and whether the mic opens.

Run this instead of guessing. It answers three questions in order, and stops at the first one that
fails, because a later answer is meaningless if an earlier one is no.

    python3 dikte/diagnose.py
"""
import sys
import time


def check_accessibility():
    from ApplicationServices import AXIsProcessTrusted
    trusted = bool(AXIsProcessTrusted())
    print(f'1. Accessibility granted to this process: {trusted}')
    if not trusted:
        print('   Without it the key listener receives nothing. Grant it to the app running this,')
        print('   then quit that app completely and reopen it.')
    return trusted


def check_keys(seconds=12):
    """Print every key event, so we can see what the keyboard actually reports."""
    from pynput import keyboard

    print(f'\n2. Press keys for {seconds} seconds. Try the key you want to dictate with,')
    print('   and hold it for a second before letting go.')
    seen = []

    def name(key):
        return getattr(key, 'name', None) or repr(getattr(key, 'char', key))

    def press(key):
        label = name(key)
        if not seen or seen[-1] != ('down', label):
            seen.append(('down', label))
            print(f'   down  {label}')

    def release(key):
        seen.append(('up', name(key)))
        print(f'   up    {name(key)}')

    listener = keyboard.Listener(on_press=press, on_release=release)
    listener.start()
    time.sleep(seconds)
    listener.stop()

    if not seen:
        print('   NOTHING was seen. The listener is not receiving events at all.')
        return None
    modifiers = [label for _, label in seen if 'cmd' in label or 'ctrl' in label
                 or 'alt' in label or 'shift' in label]
    if modifiers:
        print(f'\n   modifier keys seen: {sorted(set(modifiers))}')
    else:
        print('\n   No modifier key was seen. If you held one, this keyboard reports it differently,')
        print('   or another app is swallowing it.')
    return sorted(set(modifiers))


def check_microphone(seconds=2):
    """Open the mic exactly the way dikte does, and say what comes back."""
    import numpy as np
    import sounddevice as sd

    print(f'\n3. Opening the microphone for {seconds} seconds. Say something.')
    try:
        print(f'   default input: {sd.query_devices(sd.default.device[0])["name"]}')
    except Exception as error:
        print(f'   cannot read the default input device: {error}')
        return False

    frames = []
    try:
        with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16',
                               callback=lambda data, n, t, s: frames.append(bytes(data))):
            time.sleep(seconds)
    except Exception as error:
        print(f'   FAILED to open the microphone: {type(error).__name__}: {error}')
        print('   If macOS never asked for microphone access, grant it to the app running this')
        print('   (System Settings, Privacy & Security, Microphone), then reopen that app.')
        return False

    data = b''.join(frames)
    if not data:
        print('   The stream opened but delivered no audio at all.')
        return False
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    peak = float(np.abs(samples).max()) / 32768
    loudness = float(np.sqrt((samples ** 2).mean())) / 32768
    print(f'   got {len(samples) / 16000:.1f}s of audio, peak {peak:.3f}, level {loudness:.4f}')
    if peak < 0.001:
        print('   That is silence. macOS is handing over an empty stream, which is what a denied')
        print('   microphone permission looks like from here.')
        return False
    print('   Microphone works.')
    return True


def main():
    print('Checking what dictation needs, in order.\n')
    if not check_accessibility():
        sys.exit(1)
    modifiers = check_keys()
    if modifiers is None:
        sys.exit(1)
    check_microphone()
    print('\nTell Claude what this printed.')


if __name__ == '__main__':
    main()
