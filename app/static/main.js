/* The page: five tools behind one set of preset buttons, and the model lamps. */
import { $, $$, get, size, voices, refitGrilles, pollJobs } from "./common.js";
import * as rapid from "./rapid.js";
import * as transkripsyon from "./transkripsyon.js";
import * as dokiman from "./dokiman.js";
import * as li from "./li.js";
import * as api from "./api.js";

const TOOLS = { rapid, transkripsyon, dokiman, li, api };

function show() {
  const id = location.hash.slice(1) in TOOLS ? location.hash.slice(1) : "rapid";
  for (const name of Object.keys(TOOLS)) $(`#tab-${name}`).hidden = name !== id;
  $$(".tabs a").forEach((a) => {
    if (a.hash === `#${id}`) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  requestAnimationFrame(() => { refitGrilles(); TOOLS[id].shown?.(); });
}

function lamp(sel, st, title) {
  const elm = $(sel), words = { starting: "ap chaje", ready: "pare", failed: "pa demare" };
  elm.dataset.state = st; elm.querySelector("small").textContent = words[st] || st; elm.title = title || "";
}

let first = true;
async function health() {
  let wait = 5000;
  try {
    const j = await get("/api/health");
    lamp("#lamp-koute", j.asr, j.asr_error);
    lamp("#lamp-pale", j.tts, j.tts_mode === "loaded" ? "The voice model stays loaded between readings"
                            : j.tts_mode ? "Each reading starts the voice model, about 1 s slower" : "");
    $("#tk-disk").textContent = `Espas lib sou disk la: ${size(j.disk_free)}. Yon èdtan odyo pran anviwon 115 MB. ` +
      `(Free disk space; an hour of audio takes about 115 MB here.)`;
    if (first) {
      first = false;
      voices.openai = j.openai_voices || {};
      voices.set(j.voices);
      api.voiceTable();
    }
    if (j.asr === "starting" || j.tts === "starting") wait = 1500;
  } catch { wait = 3000; }
  setTimeout(health, wait);
}

/* light / dark: "auto" follows the computer; a choice is remembered in this browser */
function theme() {
  const root = document.documentElement, key = "kreyol-theme";
  const current = root.dataset.theme || "auto";          // set by the inline script in <head>
  const radio = document.querySelector(`.theme input[value="${current}"]`);
  if (radio) radio.checked = true;
  document.querySelectorAll(".theme input").forEach((input) => input.addEventListener("change", () => {
    if (input.value === "auto") delete root.dataset.theme; else root.dataset.theme = input.value;
    try { if (input.value === "auto") localStorage.removeItem(key); else localStorage.setItem(key, input.value); } catch { /* private window */ }
    requestAnimationFrame(refitGrilles);                 // a canvas keeps the colours it was painted with
  }));
}

theme();
for (const tool of Object.values(TOOLS)) tool.init();
window.addEventListener("hashchange", show);
show();
health();
pollJobs(true);
