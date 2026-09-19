"""OpenAI-compatible speech endpoints, so apps and scripts that already use OpenAI's audio API can use the
Kreyòl models on this computer instead: point them at http://127.0.0.1:<port>/v1, with any API key.

    GET  /v1/models
    POST /v1/audio/transcriptions   multipart: file, model, response_format (json|text|srt|vtt|verbose_json)
    POST /v1/audio/speech           JSON: input, voice, model, response_format (mp3|opus|aac|flac|wav|pcm), speed

The model name is not checked (whisper-1 or tts-1 work), and neither is the language: it is always Kreyòl.
"""
import json
import re

import engine
import jobs
from engine import UserError

MODELS = ["oswald-m3", "qwen3-tts-kreyol"]
# OpenAI's voice names, so apps with a fixed voice list still get a Kreyòl voice: women's voices map to the
# three women (f1, f2, f3), men's to the two men (m1, v5).
OPENAI_VOICES = {"alloy": "kreyol_f1", "marin": "kreyol_f1", "nova": "kreyol_f2", "coral": "kreyol_f2",
                 "shimmer": "kreyol_f3", "sage": "kreyol_f3", "echo": "kreyol_m1", "ash": "kreyol_m1",
                 "fable": "kreyol_m1", "cedar": "kreyol_m1", "onyx": "kreyol_v5", "ballad": "kreyol_v5",
                 "verse": "kreyol_v5"}
MAX_AUDIO_S = 3 * 3600
MAX_TEXT = 4096                  # OpenAI's limit; longer text belongs in the Dokiman tab
TRANSCRIPT_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")


class ApiError(Exception):
    def __init__(self, status, message, param=None, kind="invalid_request_error"):
        super().__init__(message)
        self.status, self.param, self.kind = status, param, kind

    def body(self):
        return {"error": {"message": str(self), "type": self.kind, "param": self.param, "code": None}}


def from_user_error(e):
    loading = "poko pare" in str(e)
    return ApiError(503 if loading else 400, str(e), kind="server_error" if loading else "invalid_request_error")


def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "kreyol-offline"}
                                        for m in MODELS]}


def parse_multipart(body, ctype):
    """multipart/form-data -> ({field: [values]}, {field: (filename, bytes)})."""
    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if not m:
        raise ApiError(400, "Send the request as multipart/form-data.")
    fields, files = {}, {}
    for part in body.split(b"--" + m.group(1).encode())[1:]:
        if part.startswith(b"--"):
            break
        head, _, data = part.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        disp = next((line for line in head.decode("utf-8", "replace").split("\r\n")
                     if line.lower().startswith("content-disposition")), "")
        params = dict(re.findall(r'(\w+\*?)="?([^";]*)"?', disp))
        name = params.get("name")
        if not name:
            continue
        if "filename" in params or "filename*" in params:
            files[name] = (params.get("filename") or "audio", data)
        else:
            fields.setdefault(name, []).append(data.decode("utf-8", "replace"))
    return fields, files


def transcriptions(body, ctype):
    """Returns (status, content type, body)."""
    if "multipart/form-data" not in (ctype or ""):
        raise ApiError(400, "Send the audio as multipart/form-data with a 'file' field.")
    fields, files = parse_multipart(body, ctype)
    if "file" not in files:
        raise ApiError(400, "The 'file' field is missing.", "file")
    fmt = (fields.get("response_format") or ["json"])[0]
    if fmt not in TRANSCRIPT_FORMATS:
        raise ApiError(400, f"response_format must be one of {', '.join(TRANSCRIPT_FORMATS)}.", "response_format")
    try:
        res = engine.transcribe_bytes(files["file"][1], max_s=MAX_AUDIO_S)
    except UserError as e:
        raise from_user_error(e)
    segs = res["segments"]
    if fmt == "json":
        return 200, "application/json", {"text": res["text"]}
    if fmt == "text":
        return 200, "text/plain; charset=utf-8", (res["text"] + "\n").encode()
    if fmt in ("srt", "vtt"):
        cues = jobs.cues(segs)
        if fmt == "srt":
            out = "".join(f"{i}\n{jobs.stamp(a, ',')} --> {jobs.stamp(b, ',')}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1))
        else:
            out = "WEBVTT\n\n" + "".join(f"{jobs.stamp(a, '.')} --> {jobs.stamp(b, '.')}\n{t}\n\n" for a, b, t in cues)
        return 200, "text/plain; charset=utf-8", out.encode()
    return 200, "application/json", {
        "task": "transcribe", "language": "haitian creole", "duration": res["audio_seconds"], "text": res["text"],
        "segments": [{"id": i, "seek": 0, "start": s["start"], "end": s["end"], "text": " " + s["text"],
                      "tokens": [], "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 0.0,
                      "no_speech_prob": 0.0} for i, s in enumerate(segs)]}


def pick_voice(v):
    v = (v or "alloy").strip() if isinstance(v, str) else "alloy"
    ours = dict(engine.VOICES)
    if v in ours or v in engine.custom_voices:
        return v
    for vid, label in engine.VOICES:
        if v.lower() == label.lower():
            return vid
    if v.lower() in OPENAI_VOICES:
        return OPENAI_VOICES[v.lower()]
    raise ApiError(400, f"Unknown voice '{v}'. Use one of {', '.join(ours)} or an OpenAI voice name "
                        f"({', '.join(OPENAI_VOICES)}).", "voice")


def speech(body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "The body must be JSON.")
    if not isinstance(req, dict):
        raise ApiError(400, "The body must be a JSON object.")
    text = req.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "The 'input' field is missing.", "input")
    if len(text) > MAX_TEXT:
        raise ApiError(400, f"'input' is limited to {MAX_TEXT} characters; use the Dokiman tab for longer text.", "input")
    voice = pick_voice(req.get("voice"))
    fmt = req.get("response_format") or "mp3"
    if fmt not in engine.FORMATS:
        raise ApiError(400, f"response_format must be one of {', '.join(engine.FORMATS)}.", "response_format")
    try:
        speed = float(req.get("speed", 1.0))
    except (TypeError, ValueError):
        raise ApiError(400, "speed must be a number.", "speed")
    if not 0.25 <= speed <= 4.0:
        raise ApiError(400, "speed must be between 0.25 and 4.0.", "speed")
    try:
        audio, _, _ = engine.speak(text, voice, MAX_TEXT)
    except UserError as e:
        raise from_user_error(e)
    return 200, engine.FORMATS[fmt][1], engine.encode(audio, fmt, speed)
