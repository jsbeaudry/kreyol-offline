"""Work that outlives a request: long recordings and whole folders to text, documents to audio.

Each job is a folder, travay/<id>/, with a job.json. Jobs write their results as they go, so a stopped or
interrupted job resumes where it left off. One worker per model runs that model's queue; the quick tools on
the page use the same models and slip in between two pieces of a job.
"""
import collections
import csv
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import uuid
import xml.etree.ElementTree as ET
import zipfile

import soundfile as sf

import asr_normalize
import engine
from engine import UserError

TRAVAY = os.path.join(engine.ROOT, "travay")
EXPORTS = os.path.join(TRAVAY, "_ekspo")
MEDIA_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".caf",
             ".amr", ".3gp", ".webm", ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".mpg", ".mpeg", ".ts"}
DOC_EXT = {".txt", ".md", ".docx", ".csv"}
MIN_FREE = 1 << 30            # never fill the disk: keep 1 GB free
MAX_DOC_CHARS = 100_000       # about 2 h 45 min of speech
MAX_ROWS = 2000
MAX_FOLDER_FILES = 500
PAUSE_SENTENCE, PAUSE_PARAGRAPH = 0.25, 0.8
CHARS_PER_SECOND = 10         # the five reference clips: 89 characters in 7.8-9.0 s


class NotFound(Exception):
    pass


class Stopped(Exception):
    pass


def gb(n):
    return f"{n / 1e9:.1f} GB"


def free_bytes():
    return shutil.disk_usage(TRAVAY).free


def check_space(need):
    free = free_bytes()
    if free - need < MIN_FREE:
        raise UserError(f"Pa gen ase plas sou disk la: travay sa a bezwen {gb(need)}, gen {gb(free)} lib, "
                        f"e fò {gb(MIN_FREE)} rete lib. (Not enough disk space: this needs {gb(need)}, "
                        f"{gb(free)} is free, and {gb(MIN_FREE)} must stay free.)")


def slug(name, n=40):
    s = unicodedata.normalize("NFKD", os.path.splitext(name)[0])
    s = re.sub(r"[^A-Za-z0-9]+", "-", "".join(c for c in s if not unicodedata.combining(c))).strip("-").lower()
    return s[:n] or "odyo"


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------- the job store ----------

class Store:
    def __init__(self):
        os.makedirs(TRAVAY, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs, self.cancel = {}, set()
        self.transcript_locks = collections.defaultdict(threading.Lock)
        for name in os.listdir(TRAVAY):
            path = os.path.join(TRAVAY, name, "job.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    job = json.load(f)
            except ValueError:
                continue
            if job.get("status") in ("queued", "running"):     # the server stopped while it worked
                job.update(status="stopped", eta=None,
                           detail="Sèvè a te fèmen anvan li fini. (The server stopped before this finished.)")
                write_json(path, job)
            self.jobs[job["id"]] = job

    def dir(self, job, *parts):
        return os.path.join(TRAVAY, job["id"], *parts)

    def create(self, kind, name, **fields):
        with self.lock:
            jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            job = {"id": jid, "kind": kind, "name": name, "status": "new", "progress": 0.0, "detail": "",
                   "error": "", "created": time.time(), **fields}
            os.makedirs(os.path.join(TRAVAY, jid))
            self.jobs[jid] = job
            write_json(self.dir(job, "job.json"), job)
            return job

    def update(self, job, **fields):
        with self.lock:
            job.update(fields, updated=time.time())
            if job["id"] in self.jobs:
                write_json(self.dir(job, "job.json"), job)

    def get(self, jid):
        job = self.jobs.get(jid)
        if not job:
            raise NotFound(jid)
        return job

    def list(self, kind=None):
        jobs = [j for j in self.jobs.values() if kind is None or j["kind"] == kind]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def delete(self, jid):
        job = self.get(jid)
        if job["status"] == "running":
            raise UserError("Kanpe travay la anvan ou efase l. (Stop the job before deleting it.)")
        with self.lock:
            self.cancel.add(jid)          # in case it is still waiting in a queue
            del self.jobs[jid]
            shutil.rmtree(os.path.join(TRAVAY, jid), ignore_errors=True)


store = None
queues = {"asr": queue.Queue(), "tts": queue.Queue()}
MODEL_OF = {"transkripsyon": "asr", "dokiman": "tts"}


def summary(job):
    keys = ("id", "kind", "name", "status", "progress", "detail", "error", "created", "audio_seconds",
            "speech_seconds", "took_seconds", "regions", "done", "chunks", "chars", "format", "voice_label",
            "outputs", "mode", "eta", "stats")
    out = {k: job.get(k) for k in keys}
    src = job.get("source") or {}
    out["source"] = src.get("path") if src.get("type") == "path" else None
    if out["source"]:
        home = os.path.expanduser("~")
        out["source_short"] = "~" + out["source"][len(home):] if out["source"].startswith(home + "/") else out["source"]
    return out


def enqueue(job):
    store.cancel.discard(job["id"])
    store.update(job, status="queued", error="", eta=None, detail="Ap tann tou pa l. (Waiting its turn.)")
    queues[MODEL_OF[job["kind"]]].put(job["id"])


def check(job):
    if job["id"] in store.cancel:
        raise Stopped()


def worker(model):
    q = queues[model]
    while True:
        jid = q.get()
        job = store.jobs.get(jid)
        if not job or job["status"] != "queued":
            continue
        store.update(job, status="running", detail="")
        started = time.time()
        try:
            RUNNERS[job["kind"]](job)
            store.update(job, status="done", progress=1.0, eta=None, detail="", finished=time.time(),
                         took_seconds=round(job.get("took_seconds", 0) + time.time() - started, 1))
        except Stopped:
            store.update(job, status="stopped", eta=None, detail="Ou kanpe l. (Stopped.)",
                         took_seconds=round(job.get("took_seconds", 0) + time.time() - started, 1))
        except UserError as e:
            store.update(job, status="error", eta=None, error=str(e))
        except Exception as e:
            traceback.print_exc()
            store.update(job, status="error", eta=None, error=f"{type(e).__name__}: {e}")
        finally:
            store.cancel.discard(jid)


def stop(jid):
    job = store.get(jid)
    if job["status"] == "queued":
        store.cancel.add(jid)
        store.update(job, status="stopped", detail="Ou kanpe l. (Stopped.)")
    elif job["status"] == "running":
        store.cancel.add(jid)
    return summary(job)


def resume(jid):
    job = store.get(jid)
    if job["status"] in ("stopped", "error"):
        enqueue(job)
    return summary(job)


# ---------- transcription ----------

def probe_seconds(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise UserError("Odyo sa a pa ka li: eseye yon lòt fichye. (This audio could not be read; try another file.)")


def media_ext_ok(name):
    if os.path.splitext(name)[1].lower() not in MEDIA_EXT:
        raise UserError("Se pa yon fichye odyo oswa videyo. (That is not an audio or video file.)")


def upload_target(name, size):
    """A new job for an uploaded file: returns it and where to write the upload."""
    name = os.path.basename(name or "odyo")
    media_ext_ok(name)
    check_space(size)
    ext = os.path.splitext(name)[1].lower()
    job = store.create("transkripsyon", name, source={"type": "upload", "file": "source" + ext, "size": size})
    return job, store.dir(job, "source" + ext)


def path_jobs(path, recursive=False):
    """One job per audio or video file at `path` (a file, or a folder). Files already added are skipped."""
    p = os.path.realpath(os.path.expanduser((path or "").strip()))
    if os.path.isfile(p):
        media_ext_ok(p)
        files = [p]
    elif os.path.isdir(p):
        files = []
        for root, dirs, names in os.walk(p):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".")) if recursive else []
            files += [os.path.join(root, n) for n in sorted(names, key=str.lower)
                      if not n.startswith(".") and os.path.splitext(n)[1].lower() in MEDIA_EXT]
        if not files:
            raise UserError("Pa gen odyo ni videyo nan dosye sa a. (No audio or video files in this folder.)")
    else:
        raise UserError("Chemen sa a pa egziste. (That path does not exist.)")
    if len(files) > MAX_FOLDER_FILES:
        raise UserError(f"Twòp fichye: {len(files)}; {MAX_FOLDER_FILES} maksimòm. (Too many files.)")
    known = {(j.get("source") or {}).get("path") for j in store.jobs.values()}
    new = [f for f in files if f not in known]
    jobs = [store.create("transkripsyon", os.path.basename(f), source={"type": "path", "path": f,
                                                                      "size": os.path.getsize(f)}) for f in new]
    for job in jobs:
        enqueue(job)
    return {"added": [summary(j) for j in jobs], "skipped": len(files) - len(new)}


def load_transcript(job):
    path = store.dir(job, "transcript.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_transcription(job):
    wav = store.dir(job, "audio.wav")
    src = job["source"]
    if not os.path.exists(wav):
        original = src["path"] if src["type"] == "path" else store.dir(job, src["file"])
        if not os.path.exists(original):
            raise UserError("Fichye a pa la ankò. (The file is no longer there.)")
        secs = probe_seconds(original)
        check_space(secs * engine.ASR_RATE * 2)
        store.update(job, audio_seconds=round(secs, 1), detail="Ap prepare odyo a… (Preparing the audio…)")
        part = store.dir(job, "audio.part.wav")
        secs = engine.convert(original, part, engine.ASR_RATE)
        os.replace(part, wav)
        if src["type"] == "upload":
            os.remove(original)          # the 16 kHz copy is all the page needs
        store.update(job, audio_seconds=round(secs, 2))
    check(job)

    regions_path = store.dir(job, "regions.json")
    if os.path.exists(regions_path):
        with open(regions_path) as f:
            regions = json.load(f)
    else:
        store.update(job, detail="Ap chèche kote moun ap pale… (Finding the speech…)")
        regions = engine.speech_regions(wav, job["audio_seconds"])
        write_json(regions_path, regions)
        store.update(job, regions=len(regions), done=0,
                     speech_seconds=round(sum(b - a for a, b in regions), 1))
    if not regions:
        store.update(job, stats=transcript_stats([]))
        return

    engine.wait_ready("asr", lambda: job["id"] in store.cancel)
    total = sum(b - a for a, b in regions)
    done = job.get("done", 0)
    done_s = sum(b - a for a, b in regions[:done])
    t0, s0 = time.time(), done_s
    for i in range(done, len(regions)):
        check(job)
        a, b = regions[i]
        text = engine.transcribe_region(wav, a, b)
        if text:
            with store.transcript_locks[job["id"]]:
                segs = load_transcript(job)
                segs.append({"id": i, "start": a, "end": b, "text": text, "asr_text": text, "verified": False})
                write_json(store.dir(job, "transcript.json"), segs)
        done_s += b - a
        rate = (done_s - s0) / max(time.time() - t0, 1e-6)
        store.update(job, done=i + 1, progress=done_s / total, eta=(total - done_s) / rate if rate > 0 else None,
                     detail=f"{i + 1}/{len(regions)}")
    store.update(job, stats=transcript_stats(load_transcript(job)))


def edit_distance(a, b):
    """Levenshtein distance between two sequences (words or characters)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score_segment(s):
    """How far the machine text is from the corrected text, in the form m3 was scored on."""
    ref, hyp = asr_normalize.normalize(s["text"]), asr_normalize.normalize(s["asr_text"])
    s["w_ref"], s["w_err"] = len(ref.split()), edit_distance(ref.split(), hyp.split())
    s["c_ref"], s["c_err"] = len(ref), edit_distance(ref, hyp)


def transcript_stats(segs):
    live = [s for s in segs if not s.get("deleted")]
    ver = [s for s in live if s.get("verified")]
    w_ref, w_err = sum(s.get("w_ref", 0) for s in ver), sum(s.get("w_err", 0) for s in ver)
    c_ref, c_err = sum(s.get("c_ref", 0) for s in ver), sum(s.get("c_err", 0) for s in ver)
    return {"lines": len(live), "verified": len(ver), "corrected": sum(s["text"] != s["asr_text"] for s in ver),
            "verified_seconds": round(sum(s["end"] - s["start"] for s in ver), 1),
            "wer": round(100 * w_err / w_ref, 1) if w_ref else None,
            "cer": round(100 * c_err / c_ref, 1) if c_ref else None}


def edit_segments(jid, changes):
    job = store.get(jid)
    with store.transcript_locks[jid]:
        segs = load_transcript(job)
        by_id = {s["id"]: s for s in segs}
        for c in changes:
            s = by_id.get(c.get("id"))
            if s is None:
                continue
            if "text" in c:
                s["text"] = " ".join(str(c["text"]).split())
            if "verified" in c:
                s["verified"] = bool(c["verified"])
            if "deleted" in c:
                s["deleted"] = bool(c["deleted"])
            if s.get("verified"):
                score_segment(s)
        write_json(store.dir(job, "transcript.json"), segs)
    stats = transcript_stats(segs)
    store.update(job, stats=stats)
    return stats


def job_detail(jid):
    job = store.get(jid)
    out = summary(job)
    if job["kind"] == "transkripsyon":
        segs = load_transcript(job)
        out["segments"] = [{k: s.get(k) for k in ("id", "start", "end", "text", "asr_text", "verified")}
                           for s in segs if not s.get("deleted")]
        out["stats"] = transcript_stats(segs)
        out["has_audio"] = os.path.exists(store.dir(job, "audio.wav"))
    else:
        out["files"] = sorted(os.listdir(store.dir(job, "files"))) if os.path.isdir(store.dir(job, "files")) else []
        spoken = store.dir(job, "spoken.txt")
        out["spoken"] = open(spoken, encoding="utf-8").read()[:5000] if os.path.exists(spoken) else ""
    return out


# ---------- transcript exports ----------

def stamp(t, sep):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def wrap(text, width=42):
    """Two subtitle lines at most, broken at the space nearest the middle."""
    if len(text) <= width:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    cut = min(spaces, key=lambda i: abs(i - mid)) if spaces else mid
    return text[:cut].strip() + "\n" + text[cut:].strip()


def cues(segments, max_chars=84, max_s=7.0):
    """Subtitle cues: long lines are split into pieces, timed in proportion to their length.

    The model gives no word times, so within a line the timing is an estimate.
    """
    out = []
    for s in segments:
        words, dur = s["text"].split(), s["end"] - s["start"]
        if not words:
            continue
        n = max(1, math.ceil(len(s["text"]) / max_chars), math.ceil(dur / max_s))
        n = min(n, len(words))
        groups, target, cur = [], len(s["text"]) / n, []
        for w in words:
            if cur and len(" ".join(cur + [w])) > target and len(groups) < n - 1:
                groups.append(cur)
                cur = []
            cur.append(w)
        groups.append(cur)
        total = sum(len(" ".join(g)) for g in groups)
        t = s["start"]
        for g in groups:
            text = " ".join(g)
            end = t + dur * len(text) / total
            out.append((t, end, wrap(text)))
            t = end
    return out


def live_segments(job):
    return [s for s in load_transcript(job) if not s.get("deleted") and s["text"].strip()]


def export(jid, fmt):
    job = store.get(jid)
    segs = live_segments(job)
    base = slug(job["name"], 60)
    if fmt == "srt":
        body = "".join(f"{i}\n{stamp(a, ',')} --> {stamp(b, ',')}\n{t}\n\n" for i, (a, b, t) in enumerate(cues(segs), 1))
        return body.encode(), "application/x-subrip; charset=utf-8", base + ".srt"
    if fmt == "vtt":
        body = "WEBVTT\n\n" + "".join(f"{stamp(a, '.')} --> {stamp(b, '.')}\n{t}\n\n" for a, b, t in cues(segs))
        return body.encode(), "text/vtt; charset=utf-8", base + ".vtt"
    if fmt == "txt":
        return ("\n\n".join(s["text"] for s in segs) + "\n").encode(), "text/plain; charset=utf-8", base + ".txt"
    if fmt == "json":
        body = {"name": job["name"], "audio_seconds": job.get("audio_seconds"),
                "segments": [{k: s.get(k) for k in ("start", "end", "text", "asr_text", "verified")} for s in segs]}
        return json.dumps(body, ensure_ascii=False, indent=1).encode(), "application/json", base + ".json"
    raise NotFound(fmt)


DATASET_README = """# Kreyòl speech clips

Made on {date} with the Kreyòl offline page: {rows} clips, {minutes:.1f} minutes, from {sources} recording(s).
{which}

- `clips/`: 16 kHz mono WAV, one line each (the speech region the page found, plus 0.2 s on each side).
- `metadata.csv`: `file_name`, `text` (as corrected), `text_normalized` (lowercase, no punctuation, digits
  spelled out: the form oswald-large-v3-turbo-m3 was trained on), `duration`, `source`, `start`, `end`,
  `machine_text` (what the model wrote), `verified`, `corrected`, `flags` (possible problems: digits left,
  French, an unusual speaking rate, ...).

Load it with 🤗 Datasets:

```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="path/to/this/folder")
```
"""


def export_dataset(job_ids=None, only_verified=True):
    jobs = [store.get(j) for j in job_ids] if job_ids else store.list("transkripsyon")
    items = []
    for job in jobs:
        if os.path.exists(store.dir(job, "audio.wav")):
            items += [(job, s) for s in live_segments(job) if s.get("verified") or not only_verified]
    if not items:
        raise UserError("Pa gen liy verifye pou ekspòte. Verifye kèk liy anvan. "
                        "(There are no verified lines to export yet. Check some lines first.)")
    seconds = sum(s["end"] - s["start"] for _, s in items)
    check_space(seconds * engine.ASR_RATE * 2 + 44 * len(items))
    os.makedirs(EXPORTS, exist_ok=True)
    name = f"done-kreyol-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    tmp, rows = os.path.join(EXPORTS, name + ".part"), []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for job, s in items:
            clip = f"clips/{slug(job['name'])}-{job['id'][-6:]}_{s['id']:05d}.wav"
            z.writestr(clip, engine.region_wav(store.dir(job, "audio.wav"), s["start"], s["end"]))
            dur = round(s["end"] - s["start"], 2)
            rows.append({"file_name": clip, "text": s["text"], "text_normalized": asr_normalize.normalize(s["text"]),
                         "duration": dur, "source": job["name"], "start": s["start"], "end": s["end"],
                         "machine_text": s["asr_text"], "verified": bool(s.get("verified")),
                         "corrected": s["text"] != s["asr_text"],
                         "flags": ";".join(asr_normalize.suspect(s["text"], dur))})
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        z.writestr("metadata.csv", buf.getvalue())
        z.writestr("README.md", DATASET_README.format(
            date=time.strftime("%Y-%m-%d"), rows=len(rows), minutes=seconds / 60,
            sources=len({r["source"] for r in rows}),
            which="Only lines checked by a person are included." if only_verified
            else "All lines are included, checked or not; see the `verified` column."))
    os.replace(tmp, os.path.join(EXPORTS, name))
    return {"file": name, "rows": len(rows), "seconds": round(seconds, 1),
            "size": os.path.getsize(os.path.join(EXPORTS, name))}


# ---------- documents to audio ----------

def decode_text(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


def strip_markdown(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                 # code blocks are not read aloud
    t = re.sub(r"^(\s{0,3}#{1,6}\s.*)$", r"\n\1\n", t, flags=re.M)   # a heading is its own paragraph
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                    # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)                 # links read as their text
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", t, flags=re.M)     # list markers
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    return re.sub(r"[*_`~]+", "", t)


def paragraphs(text):
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n"))]
    return [p for p in paras if p]


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_paragraphs(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        raise UserError("Dokiman Word sa a pa ka li. (This Word document could not be read.)")
    paras = []
    for p in root.iter(W + "p"):
        text = "".join(n.text or "" if n.tag == W + "t" else " " for n in p.iter()
                       if n.tag in (W + "t", W + "tab", W + "br", W + "cr"))
        if text.strip():
            paras.append(" ".join(text.split()))
    return paras


TEXT_KEYS, NAME_KEYS, VOICE_KEYS = {"text", "tèks", "teks", "texte"}, {"name", "non", "file", "fichye", "filename", "id", "nom"}, {"voice", "vwa"}


def csv_rows(text):
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return []
    head = [c.lower() for c in rows[0]]
    if TEXT_KEYS & set(head):
        ti = next(i for i, c in enumerate(head) if c in TEXT_KEYS)
        ni = next((i for i, c in enumerate(head) if c in NAME_KEYS), None)
        vi = next((i for i, c in enumerate(head) if c in VOICE_KEYS), None)
        rows = rows[1:]
    else:
        ti, ni, vi = (0, None, None) if len(rows[0]) == 1 else (1, 0, None)
    out = []
    for n, r in enumerate(rows, 1):
        text = r[ti] if ti < len(r) else ""
        if text:
            out.append({"name": (r[ni] if ni is not None and ni < len(r) and r[ni] else f"{n:04d}"),
                        "text": text, "voice": r[vi] if vi is not None and vi < len(r) else ""})
    return out


def resolve_voice(v, default):
    """A voice id or its label ("Fanm 1"), else the default."""
    v = (v or "").strip()
    for vid, label in engine.VOICES:
        if v.lower() in (vid, label.lower()):
            return vid
    return v if v in engine.custom_voices else default


def new_document(title, voice, fmt, voice_label="", text=None, filename=None, data=None):
    engine.voice_path(voice)                  # a clear error now rather than in the queue
    fmt = fmt if fmt in ("mp3", "wav") else "mp3"
    mode, name = "text", title
    if data is not None:
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in DOC_EXT:
            raise UserError("Chwazi yon fichye .txt, .md, .docx oswa .csv. (Choose a .txt, .md, .docx or .csv file.)")
        name = name or os.path.splitext(os.path.basename(filename))[0]
        if ext == ".docx":
            paras = docx_paragraphs(data)
        elif ext == ".csv":
            mode, rows = "csv", csv_rows(decode_text(data))
        else:
            text = decode_text(data)
            paras = paragraphs(strip_markdown(text) if ext == ".md" else text)
    else:
        paras = paragraphs(text or "")
    if mode == "csv":
        if len(rows) > MAX_ROWS:
            raise UserError(f"Twòp liy: {len(rows)}; {MAX_ROWS} maksimòm. (Too many rows.)")
        chunks = [{"row": i, "text": c, "voice": resolve_voice(r["voice"], voice)}
                  for i, r in enumerate(rows) for c in engine.split_text(engine.tts_normalize(r["text"]))]
        source = "\n".join(f"{r['name']}\t{r['text']}" for r in rows)
    else:
        chunks = [{"p": i, "text": c} for i, p in enumerate(paras) for c in engine.split_text(engine.tts_normalize(p))]
        source = "\n\n".join(paras)
    if not chunks:
        raise UserError("Pa gen tèks pou li. (There is no text to read.)")
    chars = sum(len(c["text"]) for c in chunks)
    if chars > MAX_DOC_CHARS:
        raise UserError(f"Dokiman an twò long: {chars} karaktè, {MAX_DOC_CHARS} maksimòm. "
                        f"(The document is too long: {chars} characters; the limit is {MAX_DOC_CHARS}.)")
    check_space(2 * chars / CHARS_PER_SECOND * engine.TTS_RATE * 2)
    if not name:
        words = paras[0].split() if mode == "text" else ["Lis"]
        name = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    job = store.create("dokiman", name, mode=mode, voice=voice, voice_label=voice_label or dict(engine.VOICES).get(voice, voice),
                       format=fmt, chunks=len(chunks), chars=chars, done=0, seed=int(time.time()) % 100000,
                       rows=[{"name": r["name"]} for r in rows] if mode == "csv" else None)
    os.makedirs(store.dir(job, "voices"))
    for v in {voice} | {c.get("voice") for c in chunks if c.get("voice")}:
        shutil.copy(engine.voice_path(v), store.dir(job, "voices", f"{v}.wav"))  # resumable after a restart
    write_json(store.dir(job, "chunks.json"), chunks)
    with open(store.dir(job, "source.txt"), "w", encoding="utf-8") as f:
        f.write(source)
    with open(store.dir(job, "spoken.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(c["text"] for c in chunks))
    enqueue(job)
    return summary(job)


def run_document(job):
    with open(store.dir(job, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    parts = store.dir(job, "parts")
    os.makedirs(parts, exist_ok=True)
    engine.wait_ready("tts", lambda: job["id"] in store.cancel)
    total = sum(len(c["text"]) for c in chunks)
    done_chars = sum(len(c["text"]) for i, c in enumerate(chunks) if os.path.exists(os.path.join(parts, f"{i:05d}.wav")))
    t0, c0 = time.time(), done_chars
    for i, c in enumerate(chunks):
        out = os.path.join(parts, f"{i:05d}.wav")
        if os.path.exists(out):
            continue
        check(job)
        ref = store.dir(job, "voices", f"{c.get('voice') or job['voice']}.wav")
        audio = engine.synth_chunk(c["text"], ref, job["seed"] + i)
        sf.write(out + ".part.wav", audio, engine.TTS_RATE, subtype="PCM_16")
        os.replace(out + ".part.wav", out)
        done_chars += len(c["text"])
        rate = (done_chars - c0) / max(time.time() - t0, 1e-6)
        store.update(job, done=i + 1, progress=0.97 * done_chars / total,
                     eta=(total - done_chars) / rate if rate > 0 else None, detail=f"{i + 1}/{len(chunks)}")
    check(job)
    store.update(job, eta=None, detail="Ap mete moso yo ansanm… (Joining the pieces…)")
    fmt = job["format"]
    if job["mode"] == "text":
        wav = store.dir(job, "audio.wav")
        audio_s = join_parts(parts, range(len(chunks)), wav,
                             [c["p"] != chunks[i - 1]["p"] if i else False for i, c in enumerate(chunks)])
        outputs = ["audio.wav"]
        if fmt == "mp3":
            engine.encode_file(wav, store.dir(job, "audio.mp3"), "mp3")
            os.remove(wav)
            outputs = ["audio.mp3"]
    else:
        files = store.dir(job, "files")
        os.makedirs(files, exist_ok=True)
        by_row = collections.OrderedDict()
        for i, c in enumerate(chunks):
            by_row.setdefault(c["row"], []).append(i)
        audio_s = 0.0
        for row, idx in by_row.items():
            base = f"{row + 1:04d}-{slug(job['rows'][row]['name'])}"
            wav = os.path.join(files, base + ".wav")
            audio_s += join_parts(parts, idx, wav, [False] * len(idx))
            if fmt == "mp3":
                engine.encode_file(wav, os.path.join(files, base + ".mp3"), "mp3")
                os.remove(wav)
        with zipfile.ZipFile(store.dir(job, "audio.zip"), "w", zipfile.ZIP_STORED) as z:
            for name in sorted(os.listdir(files)):
                z.write(os.path.join(files, name), name)
        outputs = ["audio.zip"]
    shutil.rmtree(parts, ignore_errors=True)
    store.update(job, outputs=outputs, audio_seconds=round(audio_s, 1), detail="")


def join_parts(parts, indices, dst, new_paragraph):
    """Write the pieces one after another, with a short pause between sentences and a longer one between
    paragraphs, without holding the whole recording in memory. Returns its length in seconds."""
    frames = 0
    with sf.SoundFile(dst + ".part.wav", "w", samplerate=engine.TTS_RATE, channels=1, subtype="PCM_16") as out:
        for k, i in enumerate(indices):
            if k:
                pause = engine.silence(PAUSE_PARAGRAPH if new_paragraph[k] else PAUSE_SENTENCE)
                out.write(pause)
                frames += len(pause)
            audio, _ = sf.read(os.path.join(parts, f"{i:05d}.wav"), dtype="float32")
            out.write(audio)
            frames += len(audio)
    os.replace(dst + ".part.wav", dst)
    return frames / engine.TTS_RATE


# ---------- files on this computer ----------

def browse(path):
    p = os.path.realpath(os.path.expanduser((path or "").strip() or "~"))
    if os.path.isfile(p):
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        raise UserError("Dosye sa a pa egziste. (That folder does not exist.)")
    try:
        names = sorted(os.listdir(p), key=str.lower)
    except PermissionError:
        raise UserError("Pa gen pèmisyon pou li dosye sa a. Sou Mac, bay Terminal aksè nan Réglages Système > "
                        "Confidentialité. (No permission to read this folder: on a Mac, allow Terminal in "
                        "System Settings > Privacy & Security > Files and Folders.)")
    dirs, files = [], []
    for name in names:
        full = os.path.join(p, name)
        if name.startswith("."):
            continue
        if os.path.isdir(full):
            dirs.append(name)
        elif os.path.splitext(name)[1].lower() in MEDIA_EXT:
            try:
                files.append({"name": name, "size": os.path.getsize(full)})
            except OSError:
                pass
    known = {(j.get("source") or {}).get("path") for j in store.jobs.values()}
    for f in files:
        f["added"] = os.path.join(p, f["name"]) in known
    return {"path": p, "parent": None if p == "/" else os.path.dirname(p), "home": os.path.expanduser("~"),
            "dirs": dirs[:400], "files": files[:400]}


def reveal(jid=None):
    """Show a job's folder (or the whole work folder) in Finder."""
    path = store.dir(store.get(jid)) if jid else TRAVAY
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"path": path}


RUNNERS = {"transkripsyon": run_transcription, "dokiman": run_document}


def start():
    global store
    store = Store()
    for model in queues:
        threading.Thread(target=worker, args=(model,), daemon=True).start()
