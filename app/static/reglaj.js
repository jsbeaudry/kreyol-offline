/* Reglaj: how the tools cut speech into lines and how the voice spaces a document. */
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
    render(j.values);
  } catch (e) { fail($("#set-status"), e); }
}
