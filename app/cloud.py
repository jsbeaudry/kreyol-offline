"""Read text with a voice service over HTTPS instead of the model on this machine.

The token never reaches the browser. The page asks this server to speak; this server adds the
Authorization header and returns audio. `token()` takes the value from the environment or from the
Hugging Face CLI login and hands it straight to the request — it is never written to settings.json,
never logged, and never included in an error message, because a failure is the most likely moment for
a secret to end up in a terminal someone later pastes somewhere.

The endpoint has to answer `GET /health` and `POST /v1/audio/speech` with a WAV at 24 kHz mono. The
image in cloud/ does; Hugging Face's own llama.cpp container does not, which is why that image exists.
"""
import json
import os
import urllib.error
import urllib.request

import settings

TIMEOUT = float(os.environ.get("CLOUD_TIMEOUT", "180"))
RATE = 24000


def token():
    """The bearer token, or None. Never logged, never returned to the page."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def endpoint():
    return (settings.get("tts_endpoint") or "").rstrip("/")


def enabled():
    return settings.get("tts_engine") == "cloud" and bool(endpoint())


def why_not():
    """What is missing, for the settings page to show. Says whether a token exists, never what it is."""
    if not endpoint():
        return "no address set"
    if not token():
        return "no token: set HF_TOKEN in the terminal that starts the app, or run `hf auth login`"
    return None


def _call(path, payload=None, timeout=TIMEOUT):
    """Returns (status, content_type, body). Raises only on a transport failure."""
    key = token()
    headers = {"Content-Type": "application/json", "Accept": "audio/wav, application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(endpoint() + path, data=data, headers=headers,
                                     method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def health():
    """For the settings page: is the service reachable, and what voices does it carry."""
    missing = why_not()
    if missing:
        return {"ok": False, "why": missing, "has_token": bool(token())}
    try:
        status, _, body = _call("/health", timeout=20)
    except Exception as error:
        return {"ok": False, "why": f"{type(error).__name__}: could not reach the address",
                "has_token": bool(token())}
    if status == 401 or status == 403:
        return {"ok": False, "why": "the token was refused", "has_token": True}
    if status != 200:
        return {"ok": False, "why": f"the service answered {status}", "has_token": bool(token())}
    voices = []
    try:
        status, _, body = _call("/voices", timeout=20)
        if status == 200:
            voices = json.loads(body).get("voices", [])
    except Exception:
        pass
    return {"ok": True, "voices": voices, "has_token": True, "endpoint": endpoint()}


def message(status, ctype, body):
    """A readable reason, with nothing from the request in it."""
    text = ""
    if "json" in ctype:
        try:
            payload = json.loads(body)
            text = payload.get("error") or payload.get("message") or ""
            if isinstance(text, dict):
                text = text.get("message", "")
        except ValueError:
            pass
    if status in (401, 403):
        return "the endpoint refused the token"
    if status == 404:
        return "the address has no /v1/audio/speech route — is this the llama.cpp server image?"
    if status == 503:
        return f"the endpoint is not ready{': ' + text if text else ''}"
    return f"the endpoint answered {status}{': ' + str(text)[:160] if text else ''}"


def synth(text, voice, seed):
    """One chunk of text -> float32 samples at 24 kHz, already trimmed by the service.

    `voice` is a local reference path; only its name travels, because the service carries its own copy
    of the published voices. A voice added on this machine is not there, so the caller falls back to
    the local model rather than substituting a different speaker.
    """
    import io

    import numpy as np
    import soundfile as sf

    name = os.path.splitext(os.path.basename(voice or ""))[0]
    status, ctype, body = _call("/v1/audio/speech",
                                {"input": text, "voice": name, "seed": int(seed),
                                 "response_format": "wav"})
    if status != 200:
        raise RuntimeError(message(status, ctype, body))
    if "audio" not in ctype:
        raise RuntimeError(f"expected audio, got {ctype or 'nothing'}")
    audio, rate = sf.read(io.BytesIO(body), dtype="float32")
    if rate != RATE:
        raise RuntimeError(f"expected {RATE} Hz, got {rate}")
    return audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
