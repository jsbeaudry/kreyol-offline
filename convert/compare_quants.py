"""Compare talker quantisations on this Mac: same sentences, same voices, same ASR.

One question — does a smaller quantisation still say what it was asked to say — answered the way the
TTS models were judged before: generate, transcribe, and measure the character error against the text
the model was given.

Two differences from the pod harness, both deliberate:

  It judges with the ggml m3 through whisper-server, not the PyTorch m3 on a GPU, because that is what
  is on this machine. The two differ by 0.27% of characters over 304 clips, and every variant here is
  judged by the same one, so the comparison between variants is unaffected.

  It defaults to the 36-sentence set across two voices, 72 utterances per variant, because README.md
  measured that 24 utterances move PyTorch alone from 1.5% to 2.5% CER between seed sets. Fewer than
  72 cannot tell a real difference from the sampling.

    python3 convert/compare_quants.py Q4_K_M Q2_K
    SMALL=1 python3 convert/compare_quants.py Q4_K_M Q2_K     # 24 utterances, a smoke test only
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jiwer                                                            # noqa: E402
from common import EVAL_VOICES, SENTENCES, SENTENCES_EXTRA             # noqa: E402
from kreyol_text import normalize                                      # noqa: E402
from normalize import metric_form                                      # noqa: E402

ROOT = os.path.expanduser('~/kreyol-offline')
LLAMA_TTS = f'{ROOT}/llama.cpp/build/bin/llama-tts'
WHISPER_SERVER = f'{ROOT}/whisper.cpp/build/bin/whisper-server'
ASR_MODEL = f'{ROOT}/models/ggml-oswald-m3-q5_0.bin'
VOICES = f'{ROOT}/kreyol-tts/voices'
NAME = 'qwen3-tts-1.7b-kreyol'
MMPROJ = f'{ROOT}/kreyol-tts/mmproj-{NAME}-Q8_0.gguf'
OUT = f'{ROOT}/travay/quant-compare'
PORT = 8188
SAMPLING = ['-c', '2048', '-ngl', '99', '--temp', '0.9', '--top-k', '50', '--top-p', '1.0',
            '--repeat-penalty', '1.05']


def duration(path):
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


class Talker:
    """One llama-tts held open, fed a job per line, as app/engine.py does."""

    def __init__(self, gguf):
        self.gguf = gguf
        self.proc = None
        self.replies = queue.Queue()

    def __enter__(self):
        log = open(f'{OUT}/llama-tts.log', 'w')
        self.proc = subprocess.Popen(
            [LLAMA_TTS, '-m', self.gguf, '--mmproj', MMPROJ, *SAMPLING, '-p', '-', '-o', os.devnull],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        if self.replies.get(timeout=300) != '@@tts\tready':
            raise RuntimeError(f'{os.path.basename(self.gguf)} did not start; see {OUT}/llama-tts.log')
        return self

    def __exit__(self, *_):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith('@@tts'):
                self.replies.put(line.rstrip('\n'))
        self.replies.put(None)

    def say(self, text, speaker, out, seed=0):
        self.proc.stdin.write(f"{out}\t{speaker}\t{seed}\t{' '.join(text.split())}\n")
        self.proc.stdin.flush()
        reply = self.replies.get(timeout=600)
        if reply is None or reply.split('\t')[1] != 'ok':
            raise RuntimeError(f'job failed: {reply}')


def transcribe(paths):
    """Every clip through the local whisper-server, started once for the whole comparison."""
    log = open(f'{OUT}/whisper-server.log', 'w')
    server = subprocess.Popen([WHISPER_SERVER, '-m', ASR_MODEL, '-l', 'ht', '--host', '127.0.0.1',
                               '--port', str(PORT)], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=2)
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError('whisper-server stopped')
                time.sleep(1)
        out = []
        for i, path in enumerate(paths):
            boundary = uuid.uuid4().hex
            body = b''.join([
                f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode(),
                f'--{boundary}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0.0\r\n'.encode(),
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                f'Content-Type: audio/wav\r\n\r\n'.encode(),
                open(path, 'rb').read(), f'\r\n--{boundary}--\r\n'.encode()])
            request = urllib.request.Request(f'http://127.0.0.1:{PORT}/inference', data=body,
                                             headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
            with urllib.request.urlopen(request, timeout=300) as response:
                out.append(' '.join(json.loads(response.read()).get('text', '').split()))
            if (i + 1) % 24 == 0:
                print(f'    transcribed {i + 1}/{len(paths)}', flush=True)
        return out
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def rates(refs, hyps):
    """Corpus WER and CER in percent, both sides in the training normaliser's form."""
    pairs = [(metric_form(r), metric_form(h or '')) for r, h in zip(refs, hyps)]
    pairs = [p for p in pairs if p[0]]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {'n': len(pairs), 'wer': round(100 * jiwer.wer(r, h), 2),
            'cer': round(100 * jiwer.cer(r, h), 2)}


def main():
    variants = sys.argv[1:] or ['Q4_K_M', 'Q2_K']
    sentences = SENTENCES if os.environ.get('SMALL') else SENTENCES + SENTENCES_EXTRA
    os.makedirs(OUT, exist_ok=True)
    jobs = [(voice, i) for voice in EVAL_VOICES for i in range(len(sentences))]
    print(f'{len(variants)} variants x {len(sentences)} sentences x {len(EVAL_VOICES)} voices '
          f'= {len(jobs)} utterances each', flush=True)
    if len(jobs) < 72:
        print('NOTE: fewer than 72 utterances. README.md measured that this cannot separate a real '
              'difference from sampling noise.', flush=True)

    report = {}
    for variant in variants:
        gguf = f'{ROOT}/kreyol-tts/{NAME}-{variant}.gguf'
        if not os.path.exists(gguf):
            sys.exit(f'no such file: {gguf}')
        folder = f'{OUT}/{variant}'
        os.makedirs(folder, exist_ok=True)
        print(f'\n{variant} ({os.path.getsize(gguf) / 1e6:,.0f} MB)', flush=True)
        made, audio = 0.0, 0.0
        started = time.time()
        with Talker(gguf) as talker:
            for n, (voice, i) in enumerate(jobs, 1):
                path = f'{folder}/{voice}_{i:02d}.wav'
                t = time.time()
                talker.say(normalize(sentences[i]), f'{VOICES}/{voice}.wav', path)
                made += time.time() - t
                audio += duration(path)
                if n % 12 == 0:
                    print(f'    {n}/{len(jobs)} in {time.time() - started:.0f}s', flush=True)
        print('    transcribing', flush=True)
        hyps = transcribe([f'{folder}/{v}_{i:02d}.wav' for v, i in jobs])
        refs = [normalize(sentences[i]) for _, i in jobs]
        row = {'size_mb': round(os.path.getsize(gguf) / 1e6), 'all': rates(refs, hyps),
               'rtf': round(made / audio, 2), 'audio_seconds': round(audio, 1)}
        for voice in EVAL_VOICES:
            keep = [k for k, (v, _) in enumerate(jobs) if v == voice]
            row[voice] = rates([refs[k] for k in keep], [hyps[k] for k in keep])
        row['worst'] = sorted(((rates([refs[k]], [hyps[k]])['cer'], f'{jobs[k][0]}_{jobs[k][1]:02d}',
                                hyps[k]) for k in range(len(jobs))), reverse=True)[:3]
        report[variant] = row
        print(f'    CER {row["all"]["cer"]}  WER {row["all"]["wer"]}  {row["rtf"]}x real time', flush=True)

    json.dump(report, open(f'{OUT}/compare.json', 'w'), indent=2, ensure_ascii=False)
    print('\n| variant | size | CER | WER | ' + ' | '.join(f'CER {v}' for v in EVAL_VOICES)
          + ' | real-time factor |')
    print('|---|---|---|---|' + '---|' * len(EVAL_VOICES) + '---|')
    for variant, r in report.items():
        print(f'| {variant} | {r["size_mb"]:,} MB | {r["all"]["cer"]} | {r["all"]["wer"]} | '
              + ' | '.join(str(r[v]['cer']) for v in EVAL_VOICES) + f' | {r["rtf"]} |')
    for variant, r in report.items():
        print(f'\nworst {variant}: ' + '; '.join(f'{k} CER {c}: {h[:60]!r}' for c, k, h in r['worst']))


if __name__ == '__main__':
    main()
