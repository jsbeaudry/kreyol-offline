/* Koute · Pale: the quick tools. A recording up to 10 minutes to text, a text up to 2,000 characters to speech. */
import { $, num, clock, still, post, ticker, fail, Grille, startRecording, voices, voicePicker } from "./common.js";

let grilleKoute, grillePale;

/* ---------- Koute: speech to text ---------- */
let segments = [], stopAt = null, stopRec = null, recClock = null, listening = false;
const playerK = () => $("#player-koute"), statusK = () => $("#koute-status");

async function transcribe(blob) {
  if (listening) return;
  listening = true;
  playerK().src = URL.createObjectURL(blob); playerK().hidden = false;
  $("#koute-actions").hidden = true;
  const done = ticker(statusK(), "Ap koute…");
  try {
    const j = await post("/api/transcribe", blob);
    done(); statusK().textContent = "";
    segments = j.segments;
    const box = $("#transcript");
    box.replaceChildren();
    if (!segments.length) {
      box.innerHTML = '<p class="empty">Pa gen vwa nan odyo sa a. <em>No speech was found in this audio.</em></p>';
      return;
    }
    for (const s of segments) {
      const b = document.createElement("button");
      b.className = "seg"; b.type = "button";
      b.innerHTML = `<time>${clock(s.start)}</time><span></span>`;
      b.querySelector("span").textContent = s.text;
      b.addEventListener("click", () => { playerK().currentTime = s.start; stopAt = s.end; playerK().play(); });
      box.append(b);
    }
    $("#koute-meta").textContent = `${num(j.seconds)} s pou ${num(j.audio_seconds)} s odyo`;
    $("#koute-actions").hidden = false;
  } catch (e) { done(); fail(statusK(), e); }
  finally { listening = false; }
}

/* ---------- Pale: text to speech ---------- */
let lastAudio = null, speaking = false, pale;

async function speak() {
  if (speaking) return;             // Cmd+Enter in the text box would otherwise queue a second reading
  speaking = true;
  const text = $("#text").value.trim(), btn = $("#speak"), statusP = $("#pale-status"), playerP = $("#player-pale");
  btn.disabled = true; $("#pale-actions").hidden = true; $("#spoken").hidden = true;
  const done = ticker(statusP, "Ap li…");
  try {
    const j = await post("/api/speak", { text, voice: pale.value() }, true);
    done(); statusP.textContent = "";
    lastAudio = new Blob([Uint8Array.from(atob(j.audio), (ch) => ch.charCodeAt(0))], { type: "audio/wav" });
    const url = URL.createObjectURL(lastAudio);
    playerP.src = url; playerP.hidden = false; playerP.play().catch(() => {});
    $("#download").href = url;
    const spoken = $("#spoken");
    spoken.innerHTML = "<b>Sa vwa a li:</b> "; spoken.append(j.spoken); spoken.hidden = false;
    $("#pale-meta").textContent = `${num(j.audio_seconds)} s vwa · ${num(j.seconds)} s travay`;
    $("#pale-actions").hidden = false;
  } catch (e) { done(); fail(statusP, e); }
  btn.disabled = false; speaking = false;
}

/* your own voice */
let stopVoiceRec = null, voiceClock = null, voiceCount = 0;
async function useVoice(blob) {
  const statusP = $("#pale-status"), done = ticker(statusP, "Ap prepare vwa a…");
  try {
    const j = await post("/api/voice", blob);
    done(); statusP.textContent = "";
    voiceCount += 1;
    voices.add({ id: j.id, label: `Vwa pa w${voiceCount > 1 ? " " + voiceCount : ""} · ${num(j.seconds, 0)} s` });
    $("#maker").hidden = true; $("#add-voice").setAttribute("aria-expanded", "false");
  } catch (e) { done(); fail(statusP, e); }
}

export function init() {
  grilleKoute = new Grille($("#grille-koute"));
  grillePale = new Grille($("#grille-pale"));
  grilleKoute.follow($("#player-koute"));
  grillePale.follow($("#player-pale"));
  pale = voicePicker($('[data-voices="pale"]'));

  playerK().addEventListener("timeupdate", () => {
    const t = playerK().currentTime;
    if (stopAt !== null && t >= stopAt) { playerK().pause(); stopAt = null; }
    document.querySelectorAll("#transcript .seg").forEach((elm, i) =>
      elm.classList.toggle("active", !playerK().paused && t >= segments[i].start && t < segments[i].end));
  });
  playerK().addEventListener("pause", () => document.querySelectorAll("#transcript .seg.active").forEach((elm) => elm.classList.remove("active")));

  $("#rec").addEventListener("click", async () => {
    const btn = $("#rec");
    if (stopRec) {
      const stop = stopRec; stopRec = null; clearInterval(recClock);
      btn.setAttribute("aria-pressed", "false"); $("#rec-label").textContent = "Anrejistre"; $("#rec-clock").hidden = true;
      return transcribe(await stop());
    }
    try {
      stopRec = await startRecording(grilleKoute);
      btn.setAttribute("aria-pressed", "true"); $("#rec-label").textContent = "Kanpe";
      const t0 = performance.now(), c = $("#rec-clock"); c.hidden = false; c.textContent = "00:00.0";
      recClock = setInterval(() => { c.textContent = clock((performance.now() - t0) / 1000); }, 100);
      statusK().textContent = "";
    } catch (e) { fail(statusK(), e); }
  });
  $("#file-in").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) transcribe(f); e.target.value = ""; });
  $("#copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(segments.map((s) => s.text).join(" "));
    $("#copy").textContent = "Kopye!"; setTimeout(() => { $("#copy").textContent = "Kopye tèks la"; }, 1500);
  });

  document.querySelectorAll("#tab-rapid .examples button").forEach((b) =>
    b.addEventListener("click", () => { $("#text").value = b.textContent; $("#text").focus(); }));
  $("#speak").addEventListener("click", speak);
  $("#text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) speak(); });
  $("#preview").addEventListener("click", () => {
    if (!pale.value()) return;
    const p = $("#player-pale");
    p.src = `/voices/${pale.value()}.wav`; p.hidden = false; p.play().catch(() => {});
  });
  $("#roundtrip").addEventListener("click", () => {
    if (!lastAudio) return;
    if (matchMedia("(max-width: 900px)").matches) $("#koute-title").scrollIntoView({ behavior: still ? "auto" : "smooth" });
    transcribe(lastAudio);
  });

  $("#add-voice").addEventListener("click", () => {
    const m = $("#maker"); m.hidden = !m.hidden; $("#add-voice").setAttribute("aria-expanded", String(!m.hidden));
  });
  $("#rec-voice").addEventListener("click", async () => {
    const btn = $("#rec-voice"), c = $("#voice-clock");
    const finish = async () => {
      const stop = stopVoiceRec; stopVoiceRec = null; clearInterval(voiceClock);
      btn.setAttribute("aria-pressed", "false"); btn.querySelector("span:last-child").textContent = "Anrejistre vwa a"; c.hidden = true;
      useVoice(await stop());
    };
    if (stopVoiceRec) return finish();
    try {
      stopVoiceRec = await startRecording(grillePale);
      btn.setAttribute("aria-pressed", "true"); btn.querySelector("span:last-child").textContent = "Kanpe";
      const t0 = performance.now(); c.hidden = false;
      voiceClock = setInterval(() => {               // the model uses at most 15 s, so stop there
        const t = (performance.now() - t0) / 1000; c.textContent = clock(t);
        if (t >= 15 && stopVoiceRec) finish();
      }, 100);
    } catch (e) { fail($("#pale-status"), e); }
  });
  $("#voice-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) useVoice(f); e.target.value = ""; });
}
