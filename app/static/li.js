/* Li ak mwen: hear a sentence, read it aloud, see which words the machine heard. Nothing is saved. */
import { $, el, get, post, clock, say, fail, ticker, Grille, startRecording, voicePicker } from "./common.js";

const MAX_S = 30;                 // a practice sentence never needs more than this
let lessons = [], sentences = [], index = 0, results = [], picker, grille, stopRec = null, recTimer = null, busy = false;
const audio = el("audio");
const cache = new Map();          // "voice|sentence" -> object URL of the reading

function splitSentences(text) {
  return text.split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter((s) => /\p{L}/u.test(s));
}

function renderLessons() {
  const box = $("#li-lessons");
  const chip = (id, label, checked) => {
    const input = el("input", { type: "radio", name: "li-lesson", value: id });
    input.checked = checked;
    input.addEventListener("change", () => choose(id));
    return el("label", { class: "chip" }, input, el("span", {}, label));
  };
  box.replaceChildren(...lessons.map((l, i) => chip(l.id, l.title, i === 0)), chip("pa-w", "Tèks pa w (your own)", false));
}

function choose(id) {
  $("#li-custom").hidden = id !== "pa-w";
  if (id === "pa-w") { $("#li-custom-text").focus(); return; }
  start(lessons.find((l) => l.id === id).sentences);
}

function start(list) {
  sentences = list; results = new Array(list.length).fill(null); index = 0;
  renderDots(); show();
}

function renderDots() {
  $("#li-dots").replaceChildren(...sentences.map((s, i) => el("li", {}, el("button", {
    type: "button", "aria-label": `Fraz ${i + 1}`, "aria-current": String(i === index),
    class: results[i] == null ? "" : results[i] >= 80 ? "done" : "tried",
    title: results[i] == null ? s : `${results[i]}% · ${s}`, onclick: () => { index = i; show(); },
  }, String(i + 1)))));
}

function show() {
  const s = sentences[index];
  $("#li-count").textContent = `FRAZ ${index + 1} / ${sentences.length}`;
  $("#li-sentence").replaceChildren(...s.split(/\s+/).flatMap((w, i) => [i ? " " : "", el("span", { class: "w" }, w)]));
  $("#li-result").replaceChildren();
  $("#li-legend").hidden = true;
  $("#li-next").textContent = index + 1 < sentences.length ? "Pwochen fraz →" : "Rekòmanse leson an";
  renderDots();
}

async function listen() {
  const s = sentences[index], voice = picker.value(), key = `${voice}|${s}`, btn = $("#li-listen");
  if (busy) return;
  try {
    if (!cache.has(key)) {
      busy = true; btn.disabled = true;
      const done = ticker($("#li-result"), "Ap prepare vwa a…");
      try {
        const j = await post("/api/speak", { text: s, voice }, true);
        cache.set(key, URL.createObjectURL(new Blob([Uint8Array.from(atob(j.audio), (c) => c.charCodeAt(0))], { type: "audio/wav" })));
      } finally { done(); busy = false; btn.disabled = false; $("#li-result").textContent = ""; }
    }
    audio.src = cache.get(key);
    audio.play().catch(() => {});
  } catch (e) { fail($("#li-result"), e); }
}

async function record() {
  const btn = $("#li-rec"), label = $("#li-rec-label"), c = $("#li-clock");
  if (stopRec) {
    const stop = stopRec; stopRec = null; clearInterval(recTimer);
    btn.setAttribute("aria-pressed", "false"); label.textContent = "Li l fò"; c.hidden = true;
    return check(await stop());
  }
  if (busy) return;
  try {
    audio.pause();
    stopRec = await startRecording(grille);
    btn.setAttribute("aria-pressed", "true"); label.textContent = "Mwen fini";
    const t0 = performance.now(); c.hidden = false;
    recTimer = setInterval(() => {
      const t = (performance.now() - t0) / 1000; c.textContent = clock(t);
      if (t >= MAX_S && stopRec) record();
    }, 100);
    say($("#li-result"), "Li fraz la fò, epi peze « Mwen fini ». (Read the sentence aloud, then press the button again.)");
  } catch (e) { fail($("#li-result"), e); }
}

async function check(blob) {
  const out = $("#li-result"), s = sentences[index];
  busy = true;
  const done = ticker(out, "Ap koute…");
  try {
    const r = await post(`/api/practice/check?text=${encodeURIComponent(s)}`, blob);
    done();
    const words = [...$("#li-sentence").querySelectorAll(".w")];
    r.tokens.forEach((t, i) => { if (words[i]) words[i].className = `w ${t.status === "skip" ? "" : t.status}`; });
    results[index] = r.score;
    const cheer = r.score >= 95 ? "Bravo!" : r.score >= 80 ? "Byen fèt!" : r.score >= 50 ? "Pa mal, eseye ankò." : "Eseye ankò, dousman.";
    out.replaceChildren(
      el("b", {}, `${r.ok} / ${r.total} mo`), ` · ${cheer}`,
      el("span", { class: "heard" }, `Sa machin nan tande (what the machine heard): « ${r.heard || "…"} »`));
    $("#li-legend").hidden = false;
    renderDots();
  } catch (e) { done(); fail(out, e); }
  busy = false;
}

function next() {
  if (index + 1 < sentences.length) index += 1;
  else { results = results.map(() => null); index = 0; }
  show();
}

export function shown() { grille?.fit(); }

export async function init() {
  grille = new Grille($("#grille-li"));
  grille.follow(audio);
  picker = voicePicker($('[data-voices="li"]'));
  $("#li-listen").addEventListener("click", listen);
  $("#li-rec").addEventListener("click", record);
  $("#li-next").addEventListener("click", next);
  $("#li-custom-go").addEventListener("click", () => {
    const list = splitSentences($("#li-custom-text").value);
    if (!list.length) return fail($("#li-result"), new Error("Ekri kèk fraz anvan. (Write a few sentences first.)"));
    start(list);
  });
  try {
    lessons = await get("/api/lessons");
    renderLessons();
    start(lessons[0].sentences);
  } catch (e) { fail($("#li-result"), e); }
}
