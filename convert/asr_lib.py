"""The PyTorch m3 every converted model is judged against: fp16, greedy, the same settings as evaluate.py."""
import jiwer
import librosa
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from normalize import metric_form

M3 = "jsbeaudry/oswald-large-v3-turbo-m3"
_proc = _model = None


def read16k(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    return audio if sr == 16000 else librosa.resample(audio, orig_sr=sr, target_sr=16000)


def transcribe(paths, batch=24):
    global _proc, _model
    if _model is None:
        _proc = WhisperProcessor.from_pretrained(M3, language="ht", task="transcribe")
        _model = WhisperForConditionalGeneration.from_pretrained(M3, dtype=torch.float16).to("cuda").eval()
        _model.generation_config.forced_decoder_ids = None
    audio = [read16k(p) for p in paths]
    order = sorted(range(len(paths)), key=lambda i: len(audio[i]))
    out = [""] * len(paths)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        feats = _proc.feature_extractor([audio[i] for i in idx], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = _model.generate(feats.input_features.to("cuda", torch.float16), language="ht",
                                  task="transcribe", max_new_tokens=400)
        for i, t in zip(idx, _proc.batch_decode(ids, skip_special_tokens=True)):
            out[i] = t.strip()
    return out


def rates(refs, hyps):
    """Corpus WER and CER in percent, both sides in the training normaliser's form."""
    pairs = [(metric_form(r), metric_form(h or "")) for r, h in zip(refs, hyps)]
    pairs = [p for p in pairs if p[0]]
    if not pairs:
        return {"n": 0}
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"n": len(pairs), "wer": round(100 * jiwer.wer(r, h), 2), "cer": round(100 * jiwer.cer(r, h), 2)}


def duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate
