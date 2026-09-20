"""Kreyòl offline page: a local server for the two models in this folder.

It serves the page and its tools: quick speech to text and text to speech, long recordings and whole folders
to text (with an editor and exports), documents to audio, reading practice, and an OpenAI-compatible API
under /v1. Everything binds to 127.0.0.1, and nothing is sent anywhere else.

Requests must name this server as their Host (so a web page cannot reach it through DNS rebinding), and
anything that changes something must come from the page itself or from a program, not from another site
open in the browser. `--allow-origin` lets a web app you trust call the API.

Usage: python3 app/server.py [--port 8177] [--allow-origin http://localhost:3000 ...]
"""
import argparse
import atexit
import base64
import json
import os
import re
import shutil
import signal
import sys
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bible
import engine
import jobs
import openai_api
import practice
import settings
from engine import UserError

APP = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(APP, "index.html")
STATIC = os.path.join(APP, "static")
STATIC_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
MAX_BODY = 80 << 20           # quick tools: a 10-minute recording fits comfortably
MAX_API_BODY = 200 << 20      # the API: about 3 h of MP3
MAX_JSON = 4 << 20
QUICK_AUDIO_S = 600           # Koute; longer recordings go to Transkripsyon
QUICK_TEXT = 2000             # Pale; longer text goes to Dokiman
JOB_FILE = re.compile(r"audio\.(wav|mp3|zip)|files/[\w.-]+\.(wav|mp3)")
FILE_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".zip": "application/zip"}
PORT = 8177
ALLOWED_ORIGINS = set()


def attachment(name):
    ascii_name = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{urllib.parse.quote(name)}"


class Handler(BaseHTTPRequestHandler):
    server_version = "KreyolOffline/2"

    def log_message(self, fmt, *args):
        code = str(args[1]) if len(args) > 1 else ""
        if self.command in ("GET", "HEAD", "OPTIONS") and code[:1] in ("2", "3"):
            return                                 # the page polls; only log changes and failures
        took = time.time() - getattr(self, "t0", time.time())
        print(f"{self.command} {self.path.split('?')[0]} -> {code} ({took:.1f} s)", flush=True)

    # ---------- responses ----------

    def cors(self):
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def send(self, status, body, ctype="application/json; charset=utf-8", headers=()):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in headers:
            self.send_header(k, v)
        self.cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_file(self, path, ctype, download=None):
        """With Range support, so the browser can seek in an hour-long recording."""
        size = os.path.getsize(path)
        start, end, status = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
        if m and size:
            a, b = m.groups()
            if a:
                start, end = int(a), min(size - 1, int(b)) if b else size - 1
            elif b:
                start = max(0, size - int(b))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", attachment(download))
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # ---------- requests ----------

    def body_size(self, limit):
        size = int(self.headers.get("Content-Length") or 0)
        if size > limit:
            raise UserError(f"Fichye a twò gwo: {limit >> 20} MB maksimòm isit la. (The file is too large: "
                            f"{limit >> 20} MB at most here.)")
        return size

    def body(self, limit=MAX_BODY):
        return self.rfile.read(self.body_size(limit))

    def json_body(self):
        try:
            data = json.loads(self.body(MAX_JSON) or b"{}")
        except ValueError:
            raise UserError("Demann lan pa bon. (The request is not valid JSON.)")
        if not isinstance(data, dict):
            raise UserError("Demann lan pa bon. (The request must be a JSON object.)")
        return data

    def save_body(self, dst):
        """Stream an upload to disk in 1 MB pieces, so a large video never sits in memory."""
        left = int(self.headers.get("Content-Length") or 0)
        with open(dst, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    raise ConnectionError("the upload was cut short")
                f.write(chunk)
                left -= len(chunk)

    def own_origins(self):
        return {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"} | ALLOWED_ORIGINS

    def allowed(self):
        host = (self.headers.get("Host") or "").lower()
        # host.docker.internal: apps in Docker (Open WebUI) reach this computer under that name. No web page
        # can take it as its own origin, so it cannot be used for DNS rebinding.
        if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"host.docker.internal:{PORT}"):
            return False
        origin = self.headers.get("Origin")
        return self.command in ("GET", "HEAD") or not origin or origin in self.own_origins()

    # ---------- dispatch ----------

    def do_HEAD(self):
        self.handle_any()

    def do_GET(self):
        self.handle_any()

    def do_POST(self):
        self.handle_any()

    def do_DELETE(self):
        self.handle_any()

    def do_OPTIONS(self):
        self.t0 = time.time()
        origin = self.headers.get("Origin")
        host = (self.headers.get("Host") or "").lower()
        if origin not in ALLOWED_ORIGINS or not self.path.startswith("/v1/") or \
                host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            return self.send(403, {"error": "not allowed"})
        self.send(204, b"", headers=(("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                                     ("Access-Control-Allow-Headers", "Authorization, Content-Type"),
                                     ("Access-Control-Max-Age", "600")))

    def handle_any(self):
        self.t0 = time.time()
        url = urllib.parse.urlsplit(self.path)
        path, query = url.path, {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        if not self.allowed():
            return self.send(403, {"error": "Demann sa a pa soti nan paj la. (This request did not come from the page.)"})
        api = path.startswith("/v1/")
        try:
            handler = self.route(self.command if self.command != "HEAD" else "GET", path)
            if handler is None:
                return self.send(404, {"error": "not found"})
            fn, args = handler
            result = fn(*args, query)
            if result is not None:
                self.send(200, result)
        except openai_api.ApiError as e:
            self.send(e.status, e.body())
        except jobs.NotFound:
            self.send(404, {"error": "Travay sa a pa egziste ankò. (That job no longer exists.)"})
        except UserError as e:
            self.send(400, openai_api.ApiError(400, str(e)).body() if api else {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # the browser stopped listening (seeking in audio does this)
        except Exception as e:
            traceback.print_exc()
            msg = f"Yon erè rive: {e}. Gade tèminal la. (Something failed; see the terminal.)"
            self.send(500, openai_api.ApiError(500, msg, kind="server_error").body() if api else {"error": msg})

    def route(self, method, path):
        """(function, path arguments) for this request, or None."""
        table = {
            ("GET", r"/|/index\.html"): self.page,
            ("GET", r"/static/([\w.-]+)"): self.static,
            ("GET", r"/voices/([a-z0-9_]+)\.wav"): self.voice_clip,
            ("GET", r"/api/health"): self.health,
            ("GET", r"/api/lessons"): lambda q: practice.LESSONS,
            ("GET", r"/api/settings"): lambda q: {"values": settings.all(), "fields": settings.describe()},
            ("POST", r"/api/settings"): lambda q: {"values": self.with_json(lambda d: settings.update(d.get("values") or d))},
            ("POST", r"/api/settings/reset"): lambda q: {"values": settings.reset()},
            ("GET", r"/api/bible"): self.bible_books,
            ("GET", r"/api/bible/([a-z0-9-]+)/(\d+)"): self.bible_chapter,
            ("GET", r"/api/browse"): lambda q: jobs.browse(q.get("path")),
            ("GET", r"/api/jobs"): lambda q: [jobs.summary(j) for j in jobs.store.list(q.get("kind"))],
            ("GET", r"/api/jobs/([\w-]+)"): lambda jid, q: jobs.job_detail(jid),
            ("GET", r"/api/jobs/([\w-]+)/file/(.+)"): self.job_file,
            ("GET", r"/api/jobs/([\w-]+)/export/(srt|vtt|txt|json)"): self.job_export,
            ("GET", r"/api/exports/([\w.-]+\.zip)"): self.export_file,
            ("POST", r"/api/transcribe"): lambda q: engine.transcribe_bytes(self.body(), QUICK_AUDIO_S),
            ("POST", r"/api/speak"): self.quick_speak,
            ("POST", r"/api/voice"): lambda q: engine.add_voice(self.body()),
            ("POST", r"/api/jobs/upload"): self.upload,
            ("POST", r"/api/jobs/path"): lambda q: self.with_json(lambda d: jobs.path_jobs(d.get("path"), bool(d.get("recursive")))),
            ("POST", r"/api/jobs/document"): self.document,
            ("POST", r"/api/jobs/document-file"): self.document_file,
            ("POST", r"/api/jobs/([\w-]+)/segments"): lambda jid, q: self.with_json(lambda d: jobs.edit_segments(jid, d.get("changes") or [])),
            ("POST", r"/api/jobs/([\w-]+)/stop"): lambda jid, q: jobs.stop(jid),
            ("POST", r"/api/jobs/([\w-]+)/resume"): lambda jid, q: jobs.resume(jid),
            ("POST", r"/api/jobs/([\w-]+)/redo"): lambda jid, q: jobs.redo(jid),
            ("POST", r"/api/jobs/([\w-]+)/reveal"): lambda jid, q: jobs.reveal(jid),
            ("POST", r"/api/reveal"): lambda q: jobs.reveal(),
            ("POST", r"/api/dataset"): lambda q: self.with_json(lambda d: jobs.export_dataset(d.get("jobs"), d.get("only_verified", True))),
            ("POST", r"/api/practice/check"): lambda q: practice.check(q.get("text"), self.body()),
            ("DELETE", r"/api/jobs/([\w-]+)"): lambda jid, q: jobs.store.delete(jid) or {"deleted": jid},
            ("DELETE", r"/api/exports/([\w.-]+\.zip)"): self.delete_export,
            ("GET", r"/v1/models"): lambda q: openai_api.models(),
            ("POST", r"/v1/audio/transcriptions"): self.api_transcriptions,
            ("POST", r"/v1/audio/speech"): self.api_speech,
        }
        for (m, pattern), fn in table.items():
            match = re.fullmatch(pattern, path)
            if m == method and match:
                return fn, match.groups()
        return None

    # ---------- handlers ----------

    def with_json(self, fn):
        return fn(self.json_body())

    def page(self, q):
        self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")

    def static(self, name, q):
        path = os.path.join(STATIC, name)
        ctype = STATIC_TYPES.get(os.path.splitext(name)[1])
        if not ctype or not os.path.isfile(path):
            return self.send(404, {"error": "not found"})
        self.send(200, open(path, "rb").read(), ctype)

    def voice_clip(self, voice, q):
        try:
            self.send_file(engine.voice_path(voice), "audio/wav")
        except UserError:
            self.send(404, {"error": "voice not found"})

    def health(self, q):
        running = [j for j in jobs.store.jobs.values() if j["status"] in ("running", "queued")]
        return {"asr": engine.state["asr"], "asr_error": engine.state["asr_error"], "tts": engine.state["tts"],
                "tts_mode": engine.state["tts_mode"], "port": PORT,
                "voices": [{"id": v, "label": label} for v, label in engine.VOICES],
                "bible": bible.available(),
                "openai_voices": openai_api.OPENAI_VOICES,
                "disk_free": jobs.free_bytes(), "jobs_active": len(running)}

    def bible_books(self, q):
        if not bible.available():
            return self.send(404, {"error": "Pa gen Bib la sou òdinatè sa a. (No Bible text here; see app/make_bible.py.)"})
        return {"books": bible.books()}

    def bible_chapter(self, book, number, q):
        chapter = bible.chapter(book, int(number)) if bible.available() else None
        if chapter is None:
            return self.send(404, {"error": "Chapit sa a pa la. (No such chapter.)"})
        return chapter

    def quick_speak(self, q):
        req = self.json_body()
        audio, chunks, took = engine.speak(req.get("text"), req.get("voice"), QUICK_TEXT)
        data = engine.wav_bytes(audio, engine.TTS_RATE)
        return {"audio": base64.b64encode(data).decode(), "spoken": " ".join(chunks), "parts": len(chunks),
                "seconds": round(took, 2), "audio_seconds": round(len(audio) / engine.TTS_RATE, 2)}

    def upload(self, q):
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            raise UserError("Fichye a vid. (The file is empty.)")
        job, dst = jobs.upload_target(q.get("name"), size)
        try:
            self.save_body(dst)
        except Exception:
            jobs.store.delete(job["id"])
            raise
        jobs.enqueue(job)
        return jobs.summary(job)

    def document(self, q):
        d = self.json_body()
        return jobs.new_document(d.get("title", "").strip(), d.get("voice"), d.get("format"),
                                 d.get("voice_label", ""), text=d.get("text"))

    def document_file(self, q):
        data = self.body(MAX_JSON * 5)
        return jobs.new_document(q.get("title", "").strip(), q.get("voice"), q.get("format"),
                                 q.get("voice_label", ""), filename=q.get("name"), data=data)

    def job_file(self, jid, name, q):
        job = jobs.store.get(jid)
        path = jobs.store.dir(job, *name.split("/"))
        if not JOB_FILE.fullmatch(name) or not os.path.isfile(path):
            return self.send(404, {"error": "not found"})
        ext = os.path.splitext(name)[1]
        download = None
        if q.get("download"):
            base = jobs.slug(job["name"], 60)
            download = f"{base}{ext}" if name.startswith("audio.") else os.path.basename(name)
        self.send_file(path, FILE_TYPES[ext], download)

    def job_export(self, jid, fmt, q):
        data, ctype, name = jobs.export(jid, fmt)
        self.send(200, data, ctype, headers=(("Content-Disposition", attachment(name)),))

    def export_file(self, name, q):
        path = os.path.join(jobs.EXPORTS, name)
        if not os.path.isfile(path):
            return self.send(404, {"error": "not found"})
        self.send_file(path, "application/zip", name)

    def delete_export(self, name, q):
        path = os.path.join(jobs.EXPORTS, name)
        if os.path.isfile(path):
            os.remove(path)
        return {"deleted": name}

    def api_transcriptions(self, q):
        body = self.body(MAX_API_BODY)
        status, ctype, out = openai_api.transcriptions(body, self.headers.get("Content-Type"))
        self.send(status, out, ctype if not isinstance(out, dict) else "application/json")

    def api_speech(self, q):
        status, ctype, out = openai_api.speech(self.body(MAX_JSON))
        self.send(status, out, ctype)


def main():
    global PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8177)
    parser.add_argument("--allow-origin", action="append", default=[],
                        help="a web app origin (like http://localhost:3000) allowed to call the /v1 API")
    args = parser.parse_args()
    PORT = args.port
    engine.WHISPER_PORT = args.port + 1
    ALLOWED_ORIGINS.update(o.rstrip("/") for o in args.allow_origin)
    missing = [p for p in engine.REQUIRED if not os.path.exists(p)]
    if missing:
        sys.exit("missing (run setup.sh):\n  " + "\n  ".join(missing))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is required (brew install ffmpeg)")
    atexit.register(engine.shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    settings.PATH = os.path.join(jobs.TRAVAY, "settings.json")
    jobs.start()
    engine.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True
    print(f"Kreyòl offline page: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    print(f"OpenAI-compatible API: http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
