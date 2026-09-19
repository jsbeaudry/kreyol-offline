/* Shared by every tool on the page: fetching, audio plumbing, the speaker grille, the voice list. */

export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];
export const num = (x, d = 1) => Number(x).toFixed(d).replace(".", ",");          // Kreyòl writes a decimal comma
export const clock = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(1).padStart(4, "0")}`;
export const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function hms(t) {
  t = Math.max(0, Math.floor(t));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(s).padStart(2, "0")}`;
}
export function length(sec) {                     // 45 s, 12 min, 1 h 05 min
  if (sec == null) return "";
  if (sec < 60) return `${Math.round(sec)} s`;
  if (sec < 3600) return `${Math.round(sec / 60)} min`;
  return `${Math.floor(sec / 3600)} h ${String(Math.round((sec % 3600) / 60)).padStart(2, "0")} min`;
}
export function size(n) {
  if (n >= 1e9) return `${num(n / 1e9)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) e.append(kid);
  return e;
}

export async function api(method, url, body, isJson = false) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.body = isJson ? JSON.stringify(body) : body;
    if (isJson) opts.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error?.message || j.error || `HTTP ${r.status}`);
  return j;
}
export const get = (url) => api("GET", url);
export const post = (url, body, isJson = false) => api("POST", url, body, isJson);
export const del = (url) => api("DELETE", url);

export function download(url, name = "") {        // the server marks these as attachments
  const a = el("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
}

/* ---------- status lines ---------- */
export function ticker(elm, label) {
  const t0 = performance.now();
  elm.classList.remove("error");
  const id = setInterval(() => { elm.textContent = `${label} ${num((performance.now() - t0) / 1000)} s`; }, 100);
  return () => clearInterval(id);
}
export function fail(elm, err) {
  elm.classList.add("error");
  elm.textContent = err.name === "NotAllowedError"
    ? "Mikwofòn nan bloke: bay navigatè a pèmisyon pou l itilize l. (Allow microphone access in the browser.)"
    : err.message;
}
export function say(elm, text) { elm.classList.remove("error"); elm.textContent = text; }

/* ---------- audio plumbing ---------- */
let ac = null;
export function audioCtx() {
  ac = ac || new (window.AudioContext || window.webkitAudioContext)();
  if (ac.state === "suspended") ac.resume();
  return ac;
}
const tapped = new WeakMap();                     // an element can be routed through Web Audio only once
export function analyserFor(elm) {
  if (!tapped.has(elm)) {
    const ctx = audioCtx(), an = ctx.createAnalyser();
    an.fftSize = 1024;
    ctx.createMediaElementSource(elm).connect(an);
    an.connect(ctx.destination);
    tapped.set(elm, an);
  }
  return tapped.get(elm);
}

/* The speaker grille: dots light from the centre out with the voice, red for a voice going in
   (the microphone), ink for a voice coming out (playback). */
const grilles = [];
export class Grille {
  constructor(canvas) {
    this.c = canvas; this.g = canvas.getContext("2d"); this.an = null; this.colorVar = "--ink";
    new ResizeObserver(() => this.fit()).observe(canvas);
    grilles.push(this);
  }
  fit() {
    const r = this.c.getBoundingClientRect(), d = devicePixelRatio || 1;
    if (!r.width) return;
    this.c.width = Math.round(r.width * d); this.c.height = Math.round(r.height * d); this.draw(null);
  }
  attach(an, colorVar) {
    this.an = an; this.colorVar = colorVar;
    if (still) return;
    const data = new Uint8Array(an.frequencyBinCount);
    const tick = () => { if (this.an !== an) return; an.getByteFrequencyData(data); this.draw(data); requestAnimationFrame(tick); };
    tick();
  }
  detach() { this.an = null; this.draw(null); }
  follow(audioEl) {                               // light up while this element plays
    audioEl.addEventListener("play", () => this.attach(analyserFor(audioEl), "--ink"));
    audioEl.addEventListener("pause", () => this.detach());
    audioEl.addEventListener("ended", () => this.detach());
  }
  draw(data) {
    const { c, g } = this, d = devicePixelRatio || 1, pitch = 10 * d, rad = 1.9 * d;
    const cols = Math.max(1, Math.floor(c.width / pitch)), rows = Math.max(3, Math.floor(c.height / pitch) | 1);
    const css = getComputedStyle(document.documentElement);
    const dim = css.getPropertyValue("--grille").trim(), lit = css.getPropertyValue(this.colorVar).trim();
    const x0 = (c.width - (cols - 1) * pitch) / 2, y0 = (c.height - (rows - 1) * pitch) / 2, mid = (rows - 1) / 2;
    g.clearRect(0, 0, c.width, c.height);
    for (let i = 0; i < cols; i++) {
      let level = 0;
      if (data) {                                 // quadratic bands: most columns go to the speech range
        const n = data.length * 0.5, a = Math.floor((i / cols) ** 2 * n), b = Math.max(a + 1, Math.floor(((i + 1) / cols) ** 2 * n));
        let s = 0; for (let k = a; k < b; k++) s += data[k];
        level = Math.min(1, (s / (b - a) / 255) * 1.35);
      }
      const reach = level * (mid + 1);
      for (let j = 0; j < rows; j++) {
        g.fillStyle = Math.abs(j - mid) < reach ? lit : dim;
        g.beginPath(); g.arc(x0 + i * pitch, y0 + j * pitch, rad, 0, Math.PI * 2); g.fill();
      }
    }
  }
}
// A canvas keeps the colours it was painted with: repaint idle grilles when the system theme flips,
// and fit grilles that were hidden (in another tab) when they first appear.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => grilles.forEach((g) => { if (!g.an) g.draw(null); }));
export function refitGrilles() { grilles.forEach((g) => { if (!g.an) g.fit(); }); }

export async function startRecording(grille) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = audioCtx(), an = ctx.createAnalyser();
  an.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(an);  // not to the speakers: no feedback
  grille.attach(an, "--needle");
  const rec = new MediaRecorder(stream), parts = [];
  rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
  rec.start(250);
  return () => new Promise((resolve) => {
    rec.onstop = () => { stream.getTracks().forEach((t) => t.stop()); grille.detach(); resolve(new Blob(parts, { type: rec.mimeType || "audio/webm" })); };
    rec.stop();
  });
}

/* ---------- the voices, shared by every picker on the page ---------- */
export const voices = {
  list: [],
  openai: {},
  pickers: [],
  set(list) { this.list = list; this.render(); },
  add(v) { this.list.push(v); this.render(v.id); },
  render(select) { this.pickers.forEach((p) => p.render(select)); },
  label(id) { return this.list.find((v) => v.id === id)?.label || id; },
};
export function voicePicker(container, { onChange } = {}) {
  const name = `voice-${container.dataset.voices}`;
  const picker = {
    value() { return container.querySelector(`input[name="${name}"]:checked`)?.value; },
    render(select) {
      const current = select && container.dataset.voices === "pale" ? select : this.value();
      container.querySelectorAll(".chip").forEach((c) => c.remove());
      const add = container.querySelector(".chip-add");
      voices.list.forEach((v, i) => {
        const input = el("input", { type: "radio", name, value: v.id });
        input.checked = current ? v.id === current : i === 0;
        input.addEventListener("change", () => onChange?.(v.id));
        container.insertBefore(el("label", { class: "chip" }, input, el("span", {}, v.label)), add);
      });
      if (!this.value() && voices.list.length) container.querySelector("input").checked = true;
    },
  };
  voices.pickers.push(picker);
  picker.render();
  return picker;
}

/* ---------- icons ---------- */
const svg = (d) => `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
export const ICON = {
  play: svg('<path d="M5 3.5v9l7-4.5z" fill="currentColor" stroke="none"/>'),
  check: svg('<path d="M3.5 8.5l3 3 6-7"/>'),
  close: svg('<path d="M4 4l8 8M12 4l-8 8"/>'),
  trash: svg('<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8.5h5.6l.7-8.5"/>'),
  stop: svg('<rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" stroke="none"/>'),
  resume: svg('<path d="M13 8a5 5 0 1 1-1.5-3.6M13 3v3h-3"/>'),
  folder: svg('<path d="M2 4.5h4l1.5 1.5H14v6.5H2z"/>'),
  download: svg('<path d="M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10"/>'),
};
export function iconButton(icon, label, onclick, extra = {}) {
  return el("button", { type: "button", class: "icon", "aria-label": label, title: label, html: ICON[icon], onclick, ...extra });
}

/* ---------- jobs, polled once for every tab ---------- */
export const jobsFeed = {
  list: [],
  listeners: [],
  on(fn) { this.listeners.push(fn); fn(this.list); },
  async refresh() {
    try { this.list = await get("/api/jobs"); } catch { return; }
    this.listeners.forEach((fn) => fn(this.list));
  },
  active(kind) { return this.list.filter((j) => j.kind === kind && (j.status === "running" || j.status === "queued")); },
};
let feedTimer = null;
export function pollJobs(soon = false) {
  clearTimeout(feedTimer);
  const next = () => {
    jobsFeed.refresh().then(() => {
      const busy = jobsFeed.list.some((j) => j.status === "running" || j.status === "queued");
      feedTimer = setTimeout(next, busy ? 1500 : 10000);
    });
  };
  if (soon) next(); else feedTimer = setTimeout(next, 1500);
}

export const STATUS = {
  new: "nouvo", queued: "ap tann", running: "ap travay", done: "fini", stopped: "kanpe", error: "erè",
};
