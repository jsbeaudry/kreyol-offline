"""Check the two decisions that would be maddening to debug by hand: which key press meant what,
and how a selection is cut into blocks.

Reading is bound to Left Command, the key you press for Cmd+C, Cmd+V and Cmd+Tab all day. If that
guard is wrong you do not get an error, you get a voice reading your clipboard every time you copy
something. Blocks have a similar quality: get the growth wrong and the reading stutters, which sounds
like a slow machine rather than a bug.

    python3 dikte/test_dikte.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dikte                                                             # noqa: E402
import reader                                                            # noqa: E402

CMD_L, CMD_R, C, TAB = 'cmd_l', 'cmd_r', 'c', 'tab'


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def check_shortcuts_never_read():
    """Cmd+C, Cmd+V and Cmd+Tab must all come back as shortcuts, however long they are held."""
    clock = FakeClock()
    holds = dikte.Holds([CMD_L, CMD_R], clock=clock)

    for other, label in ((C, 'Cmd+C'), (TAB, 'Cmd+Tab')):
        holds.press(CMD_L)
        clock.tick(0.05)
        holds.press(other)                 # the letter, while Command is down
        clock.tick(2.0)                    # held far past the threshold
        holds.release(other)
        key, alone, seconds = holds.release(CMD_L)
        assert key == CMD_L and not alone, f'{label} looked like a hold'
        print(f'  {label:9} held {seconds:.2f}s -> shortcut, not a reading')

    holds.press(CMD_L)                     # a tap on its own is still not a hold
    clock.tick(0.12)
    key, alone, seconds = holds.release(CMD_L)
    assert alone and seconds < dikte.HOLD_SECONDS
    print(f'  {"tap":9} held {seconds:.2f}s -> too short to be a reading')

    holds.press(CMD_L)
    clock.tick(0.8)
    key, alone, seconds = holds.release(CMD_L)
    assert alone and seconds >= dikte.HOLD_SECONDS
    print(f'  {"hold":9} held {seconds:.2f}s -> reads the selection')


def check_dictation_key():
    """Right Command alone dictates; Right Command with anything else is a shortcut to discard."""
    clock = FakeClock()
    holds = dikte.Holds([CMD_L, CMD_R], clock=clock)

    holds.press(CMD_R)
    clock.tick(1.5)
    key, alone, _ = holds.release(CMD_R)
    assert key == CMD_R and alone
    print('  right cmd alone            -> transcribe what was recorded')

    holds.press(CMD_R)
    holds.press('q')
    clock.tick(0.3)
    holds.release('q')
    key, alone, _ = holds.release(CMD_R)
    assert key == CMD_R and not alone
    print('  right cmd + another key    -> discard the recording')

    # A key that is not watched must not disturb anything.
    assert holds.press('a') is False
    assert holds.release('a') == (None, False, 0.0)
    print('  an unwatched key           -> ignored')


def check_one_at_a_time():
    """Pressing the other hotkey while one is held must not hijack it."""
    clock = FakeClock()
    holds = dikte.Holds([CMD_L, CMD_R], clock=clock)
    holds.press(CMD_R)
    assert holds.press(CMD_L) is False, 'the second hotkey started its own hold'
    clock.tick(1.0)
    key, alone, _ = holds.release(CMD_R)
    assert key == CMD_R and not alone, 'the other hotkey should count as contamination'
    print('  both hotkeys at once       -> the first one wins, and counts as a shortcut')


def check_blocks_start_small_and_grow():
    text = ('Lekòl la ap louvri lendi maten an pou tout timoun yo. Direktè a mande paran yo pou yo '
            'vini ak kaye ak liv. Reyinyon an ap fèt nan lakou a a nevè. Tout moun dwe rive alè. '
            'Si w gen kesyon, pale ak direktè a apre reyinyon an.')
    pieces = reader.blocks(text)
    sizes = [len(piece) for piece in pieces]
    assert pieces, 'no blocks'
    assert len(pieces) > 1, 'a paragraph read as one block waits out its whole length in silence'
    # The first block is one sentence: it cannot be smaller without cutting inside a sentence, which
    # is what produced a ten-character runt. It must not be the entire paragraph either.
    assert sizes[0] < len(text) / 2, f'first block {sizes[0]} is most of the text; the sound waits'
    assert max(sizes) <= reader.BLOCK
    assert min(sizes) >= reader.MIN_BLOCK, f'a runt block: {sizes}'
    for before, after in zip(sizes, sizes[1:]):
        # Each block must be makeable while the one before it plays, or the reading stutters.
        assert after <= before * reader.GROWTH + reader.MIN_BLOCK, f'{before} then {after} jumps'
    joined = ' '.join(pieces).split()
    assert len(joined) > 30, 'blocks lost most of the text'
    print(f'  {len(text)} chars -> {sizes}, sentence-sized, no runts, none jumping')

    assert reader.blocks('') == []
    assert reader.blocks('   ') == []
    assert len(reader.blocks('Wi.')) == 1
    long_text = 'Bonjou tout moun. ' * 2000
    assert sum(len(p) for p in reader.blocks(long_text)) <= reader.MAX_CHARS + reader.BLOCK
    print(f'  empty, tiny, and a {len(long_text):,}-char selection all handled')


def check_meter():
    """Quiet must read as blank and loud as full, with nothing going backwards in between."""
    assert dikte.meter(0.0) == ' ' and dikte.meter(dikte.FLOOR) == ' ', 'a quiet room is not silent'
    assert dikte.meter(1.0) == dikte.METER[-1]
    rungs = [dikte.METER.index(dikte.meter(v))
             for v in (0.0, 0.005, 0.01, 0.03, 0.08, 0.2, 0.5, 1.0)]
    assert rungs == sorted(rungs), f'the meter goes backwards: {rungs}'
    assert len(set(rungs)) >= 5, f'ordinary speech barely moves it: {rungs}'
    # Real speech should land in the middle, not pinned at either end.
    middle = dikte.METER.index(dikte.meter(0.07))
    assert 2 <= middle <= len(dikte.METER) - 2, f'speech sits at the edge of the scale: {middle}'
    print(f'  0 to 1 ->' + ''.join(dikte.meter(v) for v in
                                   (0, 0.005, 0.01, 0.03, 0.08, 0.2, 0.5, 1.0)) + '  (speech lands mid-scale)')


def check_trim_blip():
    """The click after the speech must go; the speech must not.

    This is the bug you hear rather than see: the model leaves a short burst in the silence after a
    piece, so a reading made of several pieces beeps at every join.
    """
    import numpy as np

    rate = reader.TTS_RATE
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(int(rate * 2.0)) * 0.08).astype(np.float32)
    quiet = np.zeros(int(rate * 0.5), dtype=np.float32)
    blip = (rng.standard_normal(int(rate * 0.02)) * 0.3).astype(np.float32)   # 20 ms, loud
    more_quiet = np.zeros(int(rate * 0.4), dtype=np.float32)

    out = reader.trim_blip(np.concatenate([speech, quiet, blip, more_quiet]), rate)
    kept = len(out) / rate
    assert 2.0 <= kept <= 2.3, f'kept {kept:.2f}s: the speech or too much silence survived'
    assert float(np.abs(out[int(rate * 2.2):]).max() if len(out) > rate * 2.2 else 0) < 0.05, \
        'the blip is still in there'
    print(f'  2.0s speech + 0.5s quiet + a 20ms click -> kept {kept:.2f}s, click gone')

    # Speech that simply ends must not be cut into.
    plain = np.concatenate([speech, np.zeros(int(rate * 0.1), dtype=np.float32)])
    assert len(reader.trim_blip(plain, rate)) / rate >= 2.0, 'trimmed real speech away'
    # Two real pieces separated by a breath are both speech; the second is not a click.
    two = np.concatenate([speech, np.zeros(int(rate * 0.2), dtype=np.float32), speech])
    assert len(reader.trim_blip(two, rate)) / rate > 4.0, 'treated a second sentence as a click'
    # Degenerate input must come back unharmed rather than raising inside the producer.
    assert len(reader.trim_blip(np.zeros(10, dtype=np.float32), rate)) == 10
    assert len(reader.trim_blip(np.zeros(int(rate * 0.5), dtype=np.float32), rate)) == int(rate * 0.5)
    print('  plain speech, two sentences, and silence all survive unchanged')


def check_envelope():
    import wave
    import numpy as np

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'samples', 'j1_16k.wav')
    if not os.path.exists(path):
        print('  no sample to check the envelope against; skipped')
        return
    with wave.open(path, 'rb') as w:
        seconds = w.getnframes() / w.getframerate()
    levels = reader.envelope(path)
    assert abs(len(levels) * reader.STEP - seconds) < reader.STEP * 2, 'envelope does not match duration'
    assert max(levels) > 0.01, 'the envelope found no voice'
    assert min(levels) < max(levels) / 4, 'the envelope is flat; it is not following anything'
    # A file that is not there, or not audio, must not take the reading down with it.
    assert reader.envelope('/nonexistent.wav') == [0.0]
    assert reader.envelope(__file__) == [0.0]
    print(f'  {seconds:.1f}s of speech -> {len(levels)} steps, quietest {min(levels):.4f}, '
          f'loudest {max(levels):.4f}; a missing or non-audio file gives silence')


def main():
    print('key holds: what a press meant')
    check_shortcuts_never_read()
    check_dictation_key()
    check_one_at_a_time()
    print('\nblocks: how a selection is cut for reading')
    check_blocks_start_small_and_grow()
    print('\nthe meter that replaced the beeps')
    check_meter()
    check_envelope()
    print('\nthe click the voice model leaves behind')
    check_trim_blip()
    print('\nall checks passed')


if __name__ == '__main__':
    main()
