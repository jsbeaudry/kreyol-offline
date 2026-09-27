/* Reglaj: which models run, how speech is cut into lines, and how the voice spaces a document. */
import { $, el, get, post, say, fail } from "./common.js";

const GROUPS = [
  { title: "Transkripsyon", en: "how a recording is cut into lines",
    fields: ["vad_threshold", "vad_min_silence_ms", "merge_gap_s", "max_region_s", "pad_s"] },
  { title: "Vwa a", en: "silences when the voice reads a document",
    fields: ["sentence_pause_s", "paragraph_pause_s"] },
];
const LABELS = {
  vad_threshold: ["Nivo detèksyon vwa", "speech detection level: higher hears more silence"],
  vad_min_silence_ms: ["Silans minimòm (ms)", "shortest pause that counts as silence"],
  merge_gap_s: ["Poz ki rete nan yon liy (s)", "pauses shorter than this stay inside a line"],
  max_region_s: ["Longè maksimòm yon liy (s)", "the longest a line may be"],
  pad_s: ["Ti odyo anplis chak bò (s)", "extra audio kept on each side of a line"],
  sentence_pause_s: ["Poz ant fraz yo (s)", "silence between sentences"],
  paragraph_pause_s: ["Poz ant paragraf yo (s)", "silence between paragraphs"],
};
let fields = [];

/* ---------- which models run ---------- */
/* A model is not a number: it is a file that may or may not be here, and changing one means the
   server loads it again. So these are lists, not inputs, and saving happens on the click rather than
   with the Sove button. */
let models = [];

function option(spec, opt) {
  if (opt.rejected) {
    return el("li", { class: "model bad" }, el("span", {}, opt.label),
      el("em", { class: "meta" }, opt.rejected));
  }
  const chosen = opt.path === spec.current;
  const what = opt.local ? (opt.note || "") : `${opt.mb.toLocaleString()} MB — telechaje / download`;
  const b = el("button", { class: chosen ? "model-pick on" : "model-pick", type: "button" },
    el("span", {}, chosen ? `\u2713 ${opt.label}` : opt.label),
    what ? el("em", { class: "meta" }, what) : "");
  b.addEventListener("click", () => (opt.local ? choose(spec, opt) : download(spec, opt)));
  return el("li", { class: "model" }, b);
}

async function choose(spec, opt) {
  const st = $("#set-status");
  if (opt.path === spec.current) return;
  try {
    const j = await post("/api/settings/model", { name: spec.name, path: opt.path }, true);
    models = j.models;
    drawModels();
    say(st, `${opt.label} ap chaje… (loading; the page refuses work until it is ready.)`);
  } catch (e) { fail(st, e); }
}

async function download(spec, opt) {
  const st = $("#set-status");
  if (!confirm(`Telechaje ${opt.label}? ${opt.mb.toLocaleString()} MB soti nan ${spec.repo}.`)) return;
  try {
    await post("/api/settings/fetch", { name: spec.name, path: opt.path }, true);
    say(st, `${opt.label}: telechajman kòmanse. (Downloading; progress below.)`);
    watch(spec, opt);
  } catch (e) { fail(st, e); }
}

function watch(spec, opt) {
  const st = $("#set-status");
  const timer = setInterval(async () => {
    try {
      const h = await get("/api/health");
      const d = h.download || {};
      if (d.error) { clearInterval(timer); fail(st, new Error(d.error)); return; }
      if (d.busy) { say(st, `${d.label}: ${Math.round((d.done || 0) * 100)}%`); return; }
      clearInterval(timer);
      const j = await get("/api/settings");
      models = j.models;
      drawModels();
      say(st, `${opt.label} desann. Klike sou li pou w sèvi avè l. (Downloaded; click it to use it.)`);
    } catch (e) { clearInterval(timer); fail(st, e); }
  }, 1000);
}

function drawModels() {
  $("#set-models").replaceChildren(...models.map((spec) => el("section", { class: "set-group" },
    el("h3", {}, spec.kreyol, " ", el("em", {}, spec.title)),
    el("ul", { class: "model-list" }, spec.options.map((o) => option(spec, o))))));
}


/* ---------- local voice or a cloud one ---------- */
/* Like the model lists above, this saves on the click rather than with Sove: switching the engine
   changes what the next reading does, and a half-applied choice would be confusing. The token is not
   here and never will be — the server holds it and the page only learns whether one exists. */
let engine = { choices: [], urls: [] };

function engineRow(spec) {
  const buttons = spec.options.map((name) => {
    const on = name === spec.current;
    const label = name === "cloud"
      ? ["Nan nyaj la", "cloud: the endpoint below reads the text"]
      : ["Sou machin sa a", "local: the model on this machine"];
    const b = el("button", { class: on ? "model-pick on" : "model-pick", type: "button" },
      el("span", {}, on ? `\u2713 ${label[0]}` : label[0]), el("em", { class: "meta" }, label[1]));
    b.addEventListener("click", () => setEngine(name));
    return el("li", { class: "model" }, b);
  });
  return el("ul", { class: "model-list" }, buttons);
}

function endpointRow(spec) {
  const input = el("input", { type: "url", id: "set-tts_endpoint", placeholder: "https://…",
                              value: spec.current || "", spellcheck: "false" });
  const test = el("button", { class: "model-pick", type: "button" },
    el("span", {}, "Teste"), el("em", { class: "meta" }, "check the address answers"));
  test.addEventListener("click", () => saveEndpoint(input.value, true));
  input.addEventListener("change", () => saveEndpoint(input.value, false));
  return el("div", { class: "setting" },
    el("label", { class: "cap", for: "set-tts_endpoint" }, "Adrès sèvis vwa a ", el("em", {}, "voice service address")),
    input, test,
    el("p", { class: "meta", id: "cloud-state" }, spec.about));
}

async function setEngine(name) {
  const st = $("#set-status");
  try {
    const j = await post("/api/settings", { values: { tts_engine: name } }, true);
    engine.choices = engine.choices.map((c) => (c.name === "tts_engine" ? { ...c, current: j.values.tts_engine } : c));
    drawEngine();
    if (j.values.tts_engine === "cloud") checkCloud();
    say(st, name === "cloud"
      ? "Vwa a soti nan nyaj la kounye a. (Readings now go to the endpoint; if it fails, this machine reads instead.)"
      : "Vwa a soti sou machin sa a. (Readings are generated here.)");
  } catch (e) { fail(st, e); }
}

async function saveEndpoint(url, then_test) {
  const st = $("#set-status");
  try {
    const j = await post("/api/settings", { values: { tts_endpoint: url } }, true);
    engine.urls = engine.urls.map((u) => (u.name === "tts_endpoint" ? { ...u, current: j.values.tts_endpoint } : u));
    if (!j.values.tts_endpoint && url) {
      say(st, "Adrès la pa sove: se yon adrès https san kesyon ladan l li dwe ye. (Not saved: an https address with no query string.)");
      return;
    }
    say(st, "Adrès sove. (Address saved.)");
    if (then_test || url) checkCloud();
  } catch (e) { fail(st, e); }
}

async function checkCloud() {
  const line = $("#cloud-state");
  if (!line) return;
  line.textContent = "N ap tcheke… (checking…)";
  try {
    const j = await get("/api/settings/cloud");
    line.textContent = j.ok
      ? `Li reponn. Vwa: ${(j.voices || []).join(", ") || "pa gen youn ki nonmen"}. (Reachable${(j.voices || []).length ? "" : "; it reported no voices"}.)`
      : `Li pa mache: ${j.why}.`;
  } catch (e) { line.textContent = `Li pa mache: ${e.message || e}.`; }
}

function drawEngine() {
  const host = $("#set-engine");
  if (!host) return;
  const choice = engine.choices.find((c) => c.name === "tts_engine");
  const url = engine.urls.find((u) => u.name === "tts_endpoint");
  host.replaceChildren(el("section", { class: "set-group" },
    el("h3", {}, "Ki kote vwa a fèt ", el("em", {}, "where the voice is generated")),
    choice ? engineRow(choice) : "",
    url ? endpointRow(url) : ""));
}


function row(spec) {
  const [name, about] = LABELS[spec.name] || [spec.name, spec.about];
  const input = el("input", { type: "number", id: `set-${spec.name}`, min: spec.min, max: spec.max,
                              step: spec.step, "data-name": spec.name });
  return el("div", { class: "setting" },
    el("label", { class: "cap", for: `set-${spec.name}` }, name),
    input,
    el("p", { class: "meta" }, `${about}. `, el("em", {}, `${spec.about} Default ${spec.default}.`)));
}

function render(values) {
  $("#set-fields").replaceChildren(...GROUPS.map((group) => el("section", { class: "set-group" },
    el("h3", {}, group.title, " ", el("em", {}, group.en)),
    el("div", { class: "settings-grid" }, group.fields.map((name) => row(fields.find((f) => f.name === name)))))));
  fill(values);
}
function fill(values) {
  for (const spec of fields) {
    const input = $(`#set-${spec.name}`);
    if (input) input.value = values[spec.name];
  }
}
function read() {
  const out = {};
  for (const spec of fields) out[spec.name] = Number($(`#set-${spec.name}`).value);
  return out;
}

async function save() {
  const st = $("#set-status");
  try {
    const j = await post("/api/settings", { values: read() }, true);
    fill(j.values);
    say(st, "Sove. Nouvo transkripsyon yo ap sèvi ak sa. (Saved; new transcriptions use these.)");
  } catch (e) { fail(st, e); }
}

async function reset() {
  const st = $("#set-status");
  try {
    const j = await post("/api/settings/reset", {}, true);
    fill(j.values);
    say(st, "Tout bagay tounen jan yo te ye. (Back to the defaults.)");
  } catch (e) { fail(st, e); }
}

export async function init() {
  $("#set-save").addEventListener("click", save);
  $("#set-reset").addEventListener("click", reset);
  try {
    const j = await get("/api/settings");
    fields = j.fields;
    models = j.models || [];
    engine = { choices: j.choices || [], urls: j.urls || [] };
    render(j.values);
    drawModels();
    drawEngine();
    if (j.values.tts_engine === "cloud") checkCloud();
  } catch (e) { fail($("#set-status"), e); }
}
