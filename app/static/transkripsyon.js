/* Transkripsyon: long recordings and whole folders to text, a line-by-line editor, and exports
   (subtitles, text, and the checked lines as training data). */
import { $, el, get, post, del, download, hms, length, size, num, say, fail, ticker, Grille, iconButton,
         jobsFeed, pollJobs, STATUS } from "./common.js";

const AUTOSIZE = CSS.supports("field-sizing", "content");
let current = null;            // the job open in the editor
let detailTimer = null, saveTimer = null, stopAt = null, playingId = null, lastDeleted = null;
const rows = new Map();        // segment id -> { seg, row, ta, ok }
const pending = new Map();     // segment id -> unsaved change
let grille;

const player = () => $("#tk-player");
const status = () => $("#tk-status");

/* ---------- the list of jobs ---------- */

function subline(j) {
  if (j.status === "error") return j.error;
  if (j.status === "queued") return "ap tann tou pa l (waiting)";
  if (j.status === "running") {
    if (!j.regions) return j.detail || "ap kòmanse…";
    return `${Math.round(100 * (j.progress || 0))}%` + (j.eta != null ? ` · rete ${length(j.eta)}` : "");
  }
  const parts = [];
  if (j.audio_seconds) parts.push(hms(j.audio_seconds));
  if (j.stats) parts.push(`${j.stats.lines} liy`, `${j.stats.verified} verifye`);
  if (j.status === "stopped") parts.push(`kanpe nan ${Math.round(100 * (j.progress || 0))}%`);
  return parts.join(" · ");
}

function renderJobs(list) {
  const jobs = list.filter((j) => j.kind === "transkripsyon");
  const ul = $("#tk-jobs");
  const old = new Map([...ul.children].map((li) => [li.dataset.id, li]));
  const items = jobs.map((j) => {
    const li = old.get(j.id) || el("li", { class: "job", "data-id": j.id },
      el("i"), el("button", { type: "button", class: "name", onclick: () => open(j.id) }),
      el("span", { class: "acts" }), el("span", { class: "sub" }), el("span", { class: "bar" }, el("b")));
    li.dataset.status = j.status;
    li.setAttribute("aria-current", String(current?.id === j.id));
    const name = li.querySelector(".name");
    name.textContent = j.name; name.title = j.source || j.name;
    const sub = li.querySelector(".sub");
    sub.textContent = subline(j); sub.classList.toggle("error", j.status === "error");
    const active = j.status === "running" || j.status === "queued";
    li.querySelector(".bar").hidden = !active;
    li.querySelector(".bar b").style.width = `${Math.round(100 * (j.progress || 0))}%`;
    const acts = li.querySelector(".acts");
    if (acts.dataset.for !== j.status) {
      acts.dataset.for = j.status;
      acts.replaceChildren(...[
        active && iconButton("stop", "Kanpe (stop)", () => post(`/api/jobs/${j.id}/stop`).then(() => pollJobs(true))),
        (j.status === "stopped" || j.status === "error") && iconButton("resume", "Kontinye (resume)", () => post(`/api/jobs/${j.id}/resume`).then(() => pollJobs(true))),
        j.status !== "running" && iconButton("trash", "Efase (delete)", () => remove(j)),
      ].filter(Boolean));
    }
    return li;
  });
  ul.replaceChildren(...items);
  const n = jobsFeed.active("transkripsyon").length, badge = $("#tab-count-tk");
  badge.hidden = !n; badge.textContent = n;
  if (current) {
    const j = jobs.find((x) => x.id === current.id);
    if (!j) close();
    else if (j.status !== current.status) { current.status = j.status; refresh(); }
  }
}

async function remove(j) {
  if (!confirm(`Efase "${j.name}" ak tout koreksyon li yo? (Delete this transcription and its corrections? The original file is not touched.)`)) return;
  try {
    await del(`/api/jobs/${j.id}`);
    if (current?.id === j.id) close();
    pollJobs(true);
  } catch (e) { fail(status(), e); }
}

/* ---------- adding recordings ---------- */

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/jobs/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) say(status(), `Ap voye ${file.name}… ${Math.round((100 * e.loaded) / e.total)}%`);
    };
    xhr.onload = () => {
      let j = {};
      try { j = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status < 300) resolve(j); else reject(new Error(j.error || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("Fichye a pa t ka voye. (The upload failed.)"));
    xhr.send(file);
  });
}

async function addFiles(files) {
  files = [...files];
  if (!files.length) return;
  let last = null;
  for (const f of files) {
    try { last = await upload(f); pollJobs(true); } catch (e) { fail(status(), e); return; }
  }
  say(status(), files.length === 1 ? "Fichye a ajoute. (Added.)" : `${files.length} fichye ajoute. (${files.length} files added.)`);
  if (files.length === 1 && last?.id) open(last.id);
}

/* the folder browser: files are transcribed where they are, never copied */
let browsePath = "", browseFiles = 0;
async function browse(path) {
  const list = $("#br-list");
  try {
    const d = await get(`/api/browse?path=${encodeURIComponent(path || "")}`);
    browsePath = d.path; browseFiles = d.files.length;
    const crumbs = $("#br-crumbs");
    crumbs.replaceChildren(el("button", { type: "button", onclick: () => browse(d.home) }, "⌂"));
    let acc = "";
    for (const part of d.path.split("/").filter(Boolean)) {
      acc += "/" + part;
      const target = acc;
      crumbs.append("/", el("button", { type: "button", onclick: () => browse(target) }, part));
    }
    list.replaceChildren(
      ...d.dirs.map((name) => el("li", {}, el("button", { type: "button", class: "folder", onclick: () => browse(`${d.path}/${name}`.replace("//", "/")) }, name))),
      ...d.files.map((f) => el("li", {}, el("div", { class: "file br-file" },
        el("span", {}, f.name, " ", el("small", {}, size(f.size)), f.added ? el("em", {}, " · deja ajoute") : null),
        f.added ? null : el("button", { type: "button", class: "btn tiny quiet", onclick: () => addPath(`${d.path}/${f.name}`, false) }, "Transkri")))),
    );
    if (!d.dirs.length && !d.files.length) list.append(el("li", { class: "meta" }, "Dosye sa a vid. (No folders or audio here.)"));
    updateGo();
  } catch (e) {
    list.replaceChildren(el("li", { class: "status error" }, e.message));
  }
}
function updateGo() {
  const rec = $("#br-recursive").checked, go = $("#br-go");
  go.disabled = !rec && !browseFiles;
  go.textContent = rec ? "Transkri dosye sa a ak sou-dosye yo" : `Transkri ${browseFiles} fichye nan dosye sa a`;
}
async function addPath(path, recursive) {
  try {
    const r = await post("/api/jobs/path", { path, recursive }, true);
    $("#browser").close();
    say(status(), `${r.added.length} fichye ajoute` + (r.skipped ? `, ${r.skipped} te deja la` : "") +
        `. (${r.added.length} added${r.skipped ? `, ${r.skipped} already there` : ""}.)`);
    pollJobs(true);
    if (r.added.length === 1) open(r.added[0].id);
  } catch (e) { $("#br-list").prepend(el("li", { class: "status error" }, e.message)); }
}

/* ---------- the editor ---------- */

function close() {
  clearTimeout(detailTimer);
  current = null;
  rows.clear();
  $("#tk-lines").replaceChildren();
  player().removeAttribute("src");
  $("#tk-loaded").hidden = true; $("#tk-empty").hidden = false;
  document.querySelectorAll("#tk-jobs .job").forEach((li) => li.setAttribute("aria-current", "false"));
}

async function open(id) {
  if (current?.id === id) return;
  await flush();
  close();
  document.querySelectorAll("#tk-jobs .job").forEach((li) => li.setAttribute("aria-current", String(li.dataset.id === id)));
  let j;
  try { j = await get(`/api/jobs/${id}`); } catch (e) { fail(status(), e); return; }
  current = j;
  $("#tk-empty").hidden = true; $("#tk-loaded").hidden = false;
  if (j.has_audio) player().src = `/api/jobs/${id}/file/audio.wav`;
  renderHead();
  addSegments(j.segments);
  if (matchMedia("(max-width: 980px)").matches) $("#tk-editor").scrollIntoView({ block: "start" });
  watch();
}

async function refresh() {                  // after the job's status changes
  if (!current) return;
  const id = current.id;
  try {
    const j = await get(`/api/jobs/${id}`);
    if (current?.id !== id) return;
    const hadAudio = current.has_audio;
    Object.assign(current, j);
    if (!hadAudio && j.has_audio) player().src = `/api/jobs/${id}/file/audio.wav`;
    addSegments(j.segments.filter((s) => !rows.has(s.id) && s.id !== lastDeleted?.id));
    renderHead();
  } catch { /* the next poll will try again */ }
  watch();
}
function watch() {                          // follow a job that is still working
  clearTimeout(detailTimer);
  if (current && (current.status === "running" || current.status === "queued")) detailTimer = setTimeout(refresh, 2000);
}

function renderHead() {
  const j = current;
  $("#tk-name").textContent = j.name;
  $("#tk-meta").textContent = [j.audio_seconds ? hms(j.audio_seconds) : null, j.source_short || j.source || "fichye voye (uploaded)",
                               STATUS[j.status]].filter(Boolean).join(" · ");
  $("#tk-meta").title = j.source || "";
  const st = j.stats || { lines: rows.size, verified: 0 };
  const stats = [
    [st.lines, "liy (lines)"],
    [st.verified, "verifye (checked)"],
    [st.verified_seconds ? `${num(st.verified_seconds / 60)} min` : "0 min", "done verifye (checked audio)"],
  ];
  if (st.wer != null) stats.push([`${num(st.wer)}%`, `erè modèl la sou liy verifye yo (model word error on checked lines; CER ${num(st.cer)}%)`]);
  $("#tk-stats").replaceChildren(...stats.map(([v, label]) => el("div", {}, el("b", {}, String(v)), el("span", {}, label))));
  const running = j.status === "running" || j.status === "queued";
  const note = $("#tk-running");
  note.hidden = !running;
  note.textContent = running ? `${j.status === "queued" ? "Ap tann tou pa l." : `Ap transkri… ${Math.round(100 * (j.progress || 0))}%` +
    (j.eta != null ? `, rete ${length(j.eta)}` : "")} Liy yo parèt youn apre lòt: ou ka kòmanse korije deja. ` +
    "(Lines appear as they are transcribed; you can start correcting now.)" : "";
  const exports = $("#tk-exports");
  const has = rows.size > 0;
  exports.replaceChildren(
    ...["srt", "vtt", "txt", "json"].map((f) => el("a", { class: "btn tiny quiet", href: `/api/jobs/${j.id}/export/${f}`,
      download: "", "aria-disabled": String(!has), title: { srt: "Subtitles (SRT)", vtt: "Subtitles (WebVTT)", txt: "Plain text", json: "Everything, as JSON" }[f] }, f.toUpperCase())),
    el("button", { class: "btn tiny quiet", type: "button", title: "Find the lines again with today's settings",
                   onclick: () => redo(j) }, "Refè liy yo"),
    iconButton("folder", "Wè fichye yo nan Finder (show in Finder)", () => post(`/api/jobs/${j.id}/reveal`).catch((e) => fail(status(), e))),
  );
}

async function redo(j) {
  if (!confirm(`Refè liy yo pou "${j.name}" ak reglaj jodi a? Koreksyon ou yo ap pèdi.\n\n` +
               "(Find the lines again with today's settings? Your corrections to this transcription are lost; " +
               "the audio is kept.)")) return;
  try {
    await post(`/api/jobs/${j.id}/redo`);
    rows.clear();
    $("#tk-lines").replaceChildren();
    pollJobs(true);
    refresh();
  } catch (e) { fail(status(), e); }
}

function addSegments(segs) {
  const box = $("#tk-lines");
  const fresh = [];
  for (const s of segs) {
    if (rows.has(s.id)) continue;
    const r = makeRow(s);
    rows.set(s.id, r);
    const next = [...box.children].find((row) => Number(row.dataset.id) > s.id);
    box.insertBefore(r.row, next || null);
    fresh.push(r);
  }
  if (!AUTOSIZE) requestAnimationFrame(() => fresh.forEach((r) => fit(r.ta)));
}

const same = (a, b) => a.split(/\s+/).join(" ").trim() === b;
function mark(r) {
  r.row.classList.toggle("verified", !!r.seg.verified);
  r.row.classList.toggle("corrected", !same(r.seg.text, r.seg.asr_text));
  r.ok.setAttribute("aria-pressed", String(!!r.seg.verified));
  r.row.querySelector(".t").title = same(r.seg.text, r.seg.asr_text)
    ? "Tande liy sa a (play this line)" : `Machin nan te ekri (the model wrote): ${r.seg.asr_text}`;
}
function fit(ta) {
  if (AUTOSIZE || !ta.isConnected || !ta.offsetParent) return;
  ta.style.height = "auto"; ta.style.height = `${ta.scrollHeight}px`;
}

function makeRow(s) {
  const ta = el("textarea", { rows: 1, spellcheck: "false", "aria-label": `Liy ${hms(s.start)}` });
  if (AUTOSIZE) ta.style.fieldSizing = "content";
  ta.value = s.text;
  const r = { seg: s, ta };
  r.ok = iconButton("check", "Verifye liy sa a (check this line)", () => setVerified(r, !r.seg.verified), { class: "icon ok" });
  const t = el("button", { type: "button", class: "t", onclick: () => play(r) }, hms(s.start));
  r.row = el("div", { class: "line", "data-id": s.id }, t, ta,
             el("span", { class: "acts" }, r.ok, iconButton("trash", "Efase liy sa a (delete this line)", () => removeLine(r))));
  ta.addEventListener("input", () => { r.seg.text = ta.value; fit(ta); mark(r); queue(r.seg.id, { text: ta.value }); });
  ta.addEventListener("blur", () => flush());
  ta.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== "Escape") return;
    if (e.key === "Escape") { player().pause(); return; }
    e.preventDefault();                                   // a line is one paragraph: Enter moves instead
    if (e.altKey) return play(r);
    if (e.metaKey || e.ctrlKey) { setVerified(r, true); return go(r, 1, true); }
    go(r, e.shiftKey ? -1 : 1, false);
  });
  mark(r);
  return r;
}

function go(r, step, andPlay) {
  const list = [...$("#tk-lines").children].filter((row) => row.offsetParent);
  const target = list[list.indexOf(r.row) + step];
  if (!target) return;
  const next = rows.get(Number(target.dataset.id));
  next.ta.focus();
  next.ta.setSelectionRange(next.ta.value.length, next.ta.value.length);
  if (andPlay) play(next);
}

function setVerified(r, on) {
  r.seg.verified = on;
  mark(r);
  queue(r.seg.id, { verified: on, text: r.ta.value });
  flush();
}

function removeLine(r) {
  lastDeleted = r.seg;
  r.row.remove();
  rows.delete(r.seg.id);
  queue(r.seg.id, { deleted: true });
  flush();
  const undo = el("button", { type: "button", class: "linkish", onclick: () => {
    const s = lastDeleted; lastDeleted = null;
    if (!s || !current) return;
    queue(s.id, { deleted: false }); flush();
    addSegments([s]);
    say(status(), "");
  } }, "Anile (undo)");
  status().replaceChildren("Liy lan efase. ", undo);
}

function queue(id, change) {
  pending.set(id, { ...(pending.get(id) || {}), id, ...change });
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 800);
}
async function flush() {
  clearTimeout(saveTimer);
  if (!current || !pending.size) return;
  const id = current.id, changes = [...pending.values()];
  pending.clear();
  try {
    const stats = await post(`/api/jobs/${id}/segments`, { changes }, true);
    if (current?.id === id) { current.stats = stats; renderHead(); }
  } catch (e) {
    changes.forEach((c) => pending.set(c.id, { ...c, ...(pending.get(c.id) || {}) }));
    fail(status(), e);
  }
}

function play(r) {
  const p = player();
  if (!p.src) return;
  p.currentTime = r.seg.start; stopAt = r.seg.end;
  p.play().catch(() => {});
}
function followPlayback() {
  const p = player(), t = p.currentTime;
  if (stopAt !== null && t >= stopAt) { p.pause(); stopAt = null; }
  let now = null;
  if (!p.paused) for (const r of rows.values()) if (t >= r.seg.start && t < r.seg.end) { now = r.seg.id; break; }
  if (now === playingId) return;
  rows.get(playingId)?.row.classList.remove("playing");
  playingId = now;
  const r = rows.get(now);
  if (r) {
    r.row.classList.add("playing");
    if (document.activeElement?.tagName !== "TEXTAREA") r.row.scrollIntoView({ block: "nearest" });
  }
}

export function shown() {
  grille?.fit();
  if (!AUTOSIZE) rows.forEach((r) => fit(r.ta));
}

export function init() {
  grille = new Grille($("#grille-tk"));
  grille.follow(player());
  player().addEventListener("timeupdate", followPlayback);
  player().addEventListener("pause", followPlayback);
  jobsFeed.on(renderJobs);

  $("#tk-files").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  const drop = $("#tk-drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); });

  $("#tk-browse").addEventListener("click", () => { $("#browser").showModal(); browse(browsePath); });
  $("#br-close").addEventListener("click", () => $("#browser").close());
  $("#br-recursive").addEventListener("change", updateGo);
  $("#br-go").addEventListener("click", () => addPath(browsePath, $("#br-recursive").checked));

  $("#tk-only-open").addEventListener("change", (e) => $("#tk-lines").classList.toggle("only-open", e.target.checked));
  $("#tk-dataset").addEventListener("click", async () => {
    await flush();
    const st = $("#tk-dataset-status"), done = ticker(st, "Ap prepare done yo…");
    try {
      const d = await post("/api/dataset", { only_verified: $("#tk-only-verified").checked }, true);
      done();
      say(st, `${d.rows} klip · ${num(d.seconds / 60)} min · ${size(d.size)}: ${d.file}`);
      download(`/api/exports/${d.file}`, d.file);
    } catch (e) { done(); fail(st, e); }
  });
  $("#tk-reveal").addEventListener("click", () => post("/api/reveal").catch((e) => fail(status(), e)));
  window.addEventListener("beforeunload", () => { if (pending.size) flush(); });
}
