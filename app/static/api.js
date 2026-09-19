/* API: how other apps use the models on this computer, with examples built for this server's address. */
import { $, $$, el, num, say, fail, ticker, voices, voicePicker } from "./common.js";

let picker;

function examples(base) {
  $("#api-base").textContent = base;
  $$(".api-base-inline").forEach((c) => { c.textContent = base; });
  $$(".api-base-docker").forEach((c) => { c.textContent = base.replace(/127\.0\.0\.1|localhost/, "host.docker.internal"); });
  $("#ex-curl-stt").textContent =
`# Speech to text: an audio or video file -> text (json, text, srt, vtt, verbose_json)
curl ${base}/audio/transcriptions \\
  -F file=@entrevi.m4a -F model=whisper-1 -F response_format=srt`;
  $("#ex-curl-tts").textContent =
`# Text to speech: text -> audio (mp3, opus, aac, flac, wav, pcm)
curl ${base}/audio/speech -H "Content-Type: application/json" \\
  -d '{"model": "tts-1", "voice": "kreyol_f1", "input": "Bonjou, kijan ou ye?"}' -o bonjou.mp3`;
  $("#ex-python").textContent =
`# pip install openai
from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="kreyol")

with open("entrevi.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)

audio = client.audio.speech.create(model="tts-1", voice="kreyol_m1", input="Mèsi anpil!")
audio.write_to_file("mesi.mp3")`;
}

function voiceTable() {
  const byVoice = {};
  for (const [name, id] of Object.entries(voices.openai)) (byVoice[id] ||= []).push(name);
  $("#api-voices").replaceChildren(...voices.list.filter((v) => !v.id.startsWith("custom_")).map((v) =>
    el("tr", {}, el("td", {}, v.label), el("td", {}, el("code", {}, v.id)),
       el("td", {}, (byVoice[v.id] || []).map((n, i) => [i ? ", " : "", el("code", {}, n)])))));
}

async function trySpeak() {
  const meta = $("#try-speak-meta"), btn = $("#try-speak"), t0 = performance.now();
  btn.disabled = true;
  const done = ticker(meta, "…");
  try {
    const r = await fetch("/v1/audio/speech", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "tts-1", voice: picker.value(), input: $("#try-text").value, response_format: "mp3" }) });
    if (!r.ok) throw new Error((await r.json()).error?.message || `HTTP ${r.status}`);
    const blob = await r.blob();
    done();
    const a = $("#try-audio");
    a.src = URL.createObjectURL(blob); a.hidden = false; a.play().catch(() => {});
    say(meta, `200 · ${r.headers.get("Content-Type")} · ${num(blob.size / 1024, 0)} KB · ${num((performance.now() - t0) / 1000)} s`);
  } catch (e) { done(); fail(meta, e); }
  btn.disabled = false;
}

async function tryTranscribe(file) {
  const meta = $("#try-stt-meta"), out = $("#try-out"), t0 = performance.now();
  const form = new FormData();
  form.append("file", file); form.append("model", "whisper-1"); form.append("response_format", "verbose_json");
  const done = ticker(meta, "…");
  try {
    const r = await fetch("/v1/audio/transcriptions", { method: "POST", body: form });
    const j = await r.json();
    done();
    say(meta, `${r.status} · ${num((performance.now() - t0) / 1000)} s`);
    out.textContent = JSON.stringify(j, null, 2); out.hidden = false;
  } catch (e) { done(); fail(meta, e); }
}

export function init() {
  examples(`${location.origin}/v1`);
  picker = voicePicker($('[data-voices="api"]'));
  $("#try-speak").addEventListener("click", trySpeak);
  $("#try-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) tryTranscribe(f); e.target.value = ""; });
  $$("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    await navigator.clipboard.writeText($(b.dataset.copy).textContent);
    const was = b.textContent; b.textContent = "Kopye!"; setTimeout(() => { b.textContent = was; }, 1400);
  }));
}
export { voiceTable };
