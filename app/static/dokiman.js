/* Dokiman: a whole document read aloud into one audio file, or a CSV into one file per row. */
import { $, el, post, del, length, num, say, fail, iconButton, voices, voicePicker,
         jobsFeed, pollJobs, STATUS } from "./common.js";

const CHARS_PER_SECOND = 10;     // the five voices read about 10 characters a second (89 in 7.8-9.0 s)
const SPEED = 1.36;              // and make audio 1.36 times faster than real time on an M3 Pro
let picker, file = null, sending = false;
const player = el("audio");      // one shared player for the rows of CSV jobs

function estimate() {
  const n = $("#dk-text").value.trim().length;
  const secs = n / CHARS_PER_SECOND;
  $("#dk-count").textContent = file ? "" : n
    ? `${n.toLocaleString("fr")} karaktè · ~${length(secs)} odyo · ~${length(secs / SPEED)} travay (about ${length(secs)} of audio, ${length(secs / SPEED)} to make)`
    : "";
}

async function create() {
  if (sending) return;
  const st = $("#dk-status"), format = document.querySelector('input[name="dk-format"]:checked').value;
  const voice = picker.value(), title = $("#dk-name").value.trim(), voice_label = voices.label(voice);
  if (!file && !$("#dk-text").value.trim()) return fail(st, new Error("Kole yon tèks oswa chwazi yon fichye. (Paste some text or choose a file.)"));
  sending = true; $("#dk-go").disabled = true;
  try {
    if (file) {
      const q = new URLSearchParams({ name: file.name, voice, format, title, voice_label });
      await post(`/api/jobs/document-file?${q}`, file);
    } else {
      await post("/api/jobs/document", { title, text: $("#dk-text").value, voice, voice_label, format }, true);
    }
    say(st, "Li nan lis la: ou ka kontinye travay pandan l ap fèt. (Queued; it is made in the background.)");
    pollJobs(true);
  } catch (e) { fail(st, e); }
  sending = false; $("#dk-go").disabled = false;
}

function card(j) {
  const active = j.status === "running" || j.status === "queued";
  const facts = [j.voice_label, j.chars ? `${j.chars.toLocaleString("fr")} karaktè` : null,
                 j.audio_seconds ? length(j.audio_seconds) : null, j.mode === "csv" ? `${j.chunks} moso, CSV` : null]
    .filter(Boolean).join(" · ");
  const head = el("header", {}, el("b", { title: j.name }, j.name),
    el("span", { class: "row", style: "gap:2px;flex-wrap:nowrap" },
      el("span", { class: "meta" }, STATUS[j.status]),
      active && iconButton("stop", "Kanpe (stop)", () => post(`/api/jobs/${j.id}/stop`).then(() => pollJobs(true))),
      (j.status === "stopped" || j.status === "error") && iconButton("resume", "Kontinye (resume)", () => post(`/api/jobs/${j.id}/resume`).then(() => pollJobs(true))),
      j.status !== "running" && iconButton("trash", "Efase (delete)", async () => {
        if (!confirm(`Efase "${j.name}"? (Delete this audio?)`)) return;
        await del(`/api/jobs/${j.id}`).catch((e) => fail($("#dk-status"), e));
        pollJobs(true);
      })));
  const body = [el("p", { class: "meta", style: "margin:4px 0 0" }, facts)];
  if (active) {
    body.push(el("div", { class: "bar" }, el("b", { style: `width:${Math.round(100 * (j.progress || 0))}%` })),
              el("p", { class: "meta", style: "margin:2px 0 0" }, j.status === "queued" ? "ap tann tou pa l (waiting)"
                : `${Math.round(100 * (j.progress || 0))}%` + (j.eta != null ? ` · rete ${length(j.eta)}` : "")));
  }
  if (j.status === "error") body.push(el("p", { class: "status error" }, j.error));
  if (j.status === "done" && j.outputs?.length) {
    const out = j.outputs[0], url = `/api/jobs/${j.id}/file/${out}`;
    if (out.endsWith(".zip")) {
      body.push(el("div", { class: "doc-out" }, filesList(j),
        el("div", { class: "row" }, el("a", { class: "btn small", href: `${url}?download=1`, download: "" }, "Telechaje tout (.zip)"))));
    } else {
      const audio = el("audio", { controls: true, preload: "none", src: url });
      body.push(el("div", { class: "doc-out" }, audio,
        el("div", { class: "row" }, el("a", { class: "btn small", href: `${url}?download=1`, download: "" }, `Telechaje (${out.split(".").pop().toUpperCase()})`))));
    }
  }
  return el("article", { class: "job-card", "data-id": j.id, "data-status": j.status }, head, ...body);
}

function filesList(j) {
  const ul = el("ul", { class: "doc-files" });
  // the file names are only known once the job is done; ask for them when the card is drawn
  fetch(`/api/jobs/${j.id}`).then((r) => r.json()).then((d) => {
    for (const name of d.files || []) {
      const url = `/api/jobs/${j.id}/file/files/${name}`;
      ul.append(el("li", {},
        iconButton("play", `Tande ${name}`, () => { player.src = url; player.play().catch(() => {}); }),
        el("span", {}, name),
        el("a", { class: "btn tiny quiet", href: `${url}?download=1`, download: "" }, "Telechaje")));
    }
  });
  return ul;
}

let lastSig = "";
function render(list) {
  const jobs = list.filter((j) => j.kind === "dokiman");
  const n = jobsFeed.active("dokiman").length, badge = $("#tab-count-dk");
  badge.hidden = !n; badge.textContent = n;
  // redraw only when something changed, or playing audio would restart
  const sig = JSON.stringify(jobs.map((j) => [j.id, j.status, Math.round(100 * (j.progress || 0)), j.eta && Math.round(j.eta / 10)]));
  if (sig === lastSig) return;
  lastSig = sig;
  const box = $("#dk-jobs");
  const keep = new Map([...box.children].map((c) => [c.dataset.id, c]));
  box.replaceChildren(...jobs.map((j) => {
    const old = keep.get(j.id);
    return old && old.dataset.status === "done" && j.status === "done" ? old : card(j);
  }));
  $("#dk-empty").hidden = jobs.length > 0;
}

export function init() {
  picker = voicePicker($('[data-voices="dokiman"]'));
  $("#dk-text").addEventListener("input", estimate);
  $("#dk-file").addEventListener("change", (e) => {
    file = e.target.files[0] || null; e.target.value = "";
    $("#dk-file-name").textContent = file ? `${file.name} (${num(file.size / 1024, 0)} KB): tèks la anlè a pa konte` : ".txt, .md, .docx, .csv";
    $("#dk-file-clear").hidden = !file;
    $("#dk-text").disabled = !!file;
    estimate();
  });
  $("#dk-file-clear").addEventListener("click", () => {
    file = null; $("#dk-file-name").textContent = ".txt, .md, .docx, .csv"; $("#dk-file-clear").hidden = true;
    $("#dk-text").disabled = false; estimate();
  });
  $("#dk-go").addEventListener("click", create);
  jobsFeed.on(render);
  estimate();
}
