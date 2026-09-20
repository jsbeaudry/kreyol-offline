/* Bib la: read a chapter and hear it, line by line, from the text on this computer. */
import { $, el, get, post, length, say, fail, Grille, voicePicker, voices, pollJobs } from "./common.js";
import { practiceWith } from "./li.js";

const KEY = "kreyol-bib";
const KEEP = 16;                   // readings kept in memory; a chapter's worth of WAV is a few MB
let books = [], book = null, chapter = 1, lines = [], picker, grille;
let playing = false, current = -1;
const audio = el("audio");
const cache = new Map();           // "voice|book|chapter|line" -> object URL

const status = () => $("#bib-status");

function remember() {
  try { localStorage.setItem(KEY, JSON.stringify({ book, chapter })); } catch { /* private window */ }
}
function recall() {
  try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch { return null; }
}

/* ---------- the text ---------- */

function fillChapters() {
  const n = books.find((b) => b.id === book)?.chapters || 1;
  if (chapter > n) chapter = 1;
  $("#bib-chapter").replaceChildren(...Array.from({ length: n }, (_, i) =>
    el("option", { value: String(i + 1) }, `Chapit ${i + 1}`)));
  $("#bib-chapter").value = String(chapter);
}

async function show() {
  stop();
  try {
    const c = await get(`/api/bible/${book}/${chapter}`);
    lines = c.lines;
    $("#bib-lines").replaceChildren(...lines.map((text, i) => el("div", { class: "bib-line", "data-i": String(i) },
      el("button", { type: "button", class: "n", title: "Tande depi liy sa a (hear it from this line)", onclick: () => read(i) }, String(i + 1)),
      el("p", {}, text))));
    $("#bib-meta").textContent = `${c.name} ${c.chapter} · ${lines.length} liy · ${c.chars.toLocaleString("fr")} karaktè · ` +
      `anviwon ${length(c.chars / 10)} odyo (about ${length(c.chars / 10)} to listen to)`;
    say(status(), "");
    remember();
  } catch (e) { fail(status(), e); }
}

/* ---------- reading aloud ---------- */

function trimCache() {
  while (cache.size > KEEP) {
    const [k, url] = cache.entries().next().value;
    URL.revokeObjectURL(url);
    cache.delete(k);
  }
}
async function clip(i) {
  const voice = picker.value(), key = `${voice}|${book}|${chapter}|${i}`;
  if (!cache.has(key)) {
    const j = await post("/api/speak", { text: lines[i], voice }, true);
    cache.set(key, URL.createObjectURL(new Blob([Uint8Array.from(atob(j.audio), (c) => c.charCodeAt(0))], { type: "audio/wav" })));
    trimCache();
  }
  return cache.get(key);
}

function mark(i) {
  document.querySelectorAll("#bib-lines .bib-line.playing").forEach((row) => row.classList.remove("playing"));
  current = i;
  const row = document.querySelector(`#bib-lines .bib-line[data-i="${i}"]`);
  if (row) { row.classList.add("playing"); row.scrollIntoView({ block: "nearest" }); }
}

function play(url) {
  return new Promise((resolve) => {
    audio.src = url;
    audio.onended = audio.onerror = resolve;
    audio.play().catch(resolve);
  });
}

async function read(from) {
  if (playing) stop();
  playing = true;
  $("#bib-stop").hidden = false;
  $("#bib-read").disabled = true;
  for (let i = from; i < lines.length && playing; i++) {
    mark(i);
    let url;
    try {
      if (!cache.has(`${picker.value()}|${book}|${chapter}|${i}`)) say(status(), `Ap prepare liy ${i + 1}… (preparing line ${i + 1})`);
      url = await clip(i);
    } catch (e) { fail(status(), e); break; }
    if (!playing) break;
    say(status(), `Ap li liy ${i + 1} nan ${lines.length}. (Reading line ${i + 1} of ${lines.length}.)`);
    clip(i + 1 < lines.length ? i + 1 : i).catch(() => {});      // get the next one ready while this plays
    await play(url);
  }
  if (playing && current >= lines.length - 1) say(status(), "Fini chapit la. (End of the chapter.)");
  stop(true);
}

function stop(keepMessage) {
  playing = false;
  audio.pause();
  $("#bib-stop").hidden = true;
  $("#bib-read").disabled = false;
  document.querySelectorAll("#bib-lines .bib-line.playing").forEach((row) => row.classList.remove("playing"));
  if (!keepMessage) say(status(), "");
}

/* ---------- the whole chapter as a file, or as practice ---------- */

async function makeFile() {
  const voice = picker.value();
  try {
    await post("/api/jobs/document", {
      title: `${books.find((b) => b.id === book).name} ${chapter}`, text: lines.join(" "),
      voice, voice_label: voices.label(voice), format: "mp3",
    }, true);
    say(status(), "Y ap fè fichye a nan Dokiman: ou ka kontinye li pandan sa. " +
                  "(It is being made in the Dokiman tab; you can keep reading meanwhile.)");
    pollJobs(true);
  } catch (e) { fail(status(), e); }
}

export function shown() { grille?.fit(); }

export async function init() {
  grille = new Grille($("#grille-bib"));
  grille.follow(audio);
  picker = voicePicker($('[data-voices="bib"]'));
  $("#bib-book").addEventListener("change", (e) => { book = e.target.value; chapter = 1; fillChapters(); show(); });
  $("#bib-chapter").addEventListener("change", (e) => { chapter = Number(e.target.value); show(); });
  $("#bib-read").addEventListener("click", () => read(0));
  $("#bib-stop").addEventListener("click", () => stop());
  $("#bib-file").addEventListener("click", makeFile);
  $("#bib-practice").addEventListener("click", () => {
    stop();
    practiceWith(lines, `${books.find((b) => b.id === book).name} ${chapter}`);
  });
  $("#bib-prev").addEventListener("click", () => step(-1));
  $("#bib-next").addEventListener("click", () => step(1));
  try {
    books = (await get("/api/bible")).books;
  } catch { return; }                                   // no Bible file here: main.js hides the tab
  $("#bib-book").replaceChildren(...books.map((b) => el("option", { value: b.id }, b.name)));
  const saved = recall();
  book = books.some((b) => b.id === saved?.book) ? saved.book : books[0].id;
  chapter = Number(saved?.chapter) || 1;
  $("#bib-book").value = book;
  fillChapters();
  show();
}

function step(by) {
  const i = books.findIndex((b) => b.id === book), n = books[i].chapters;
  if (chapter + by >= 1 && chapter + by <= n) chapter += by;
  else if (by > 0 && i + 1 < books.length) { book = books[i + 1].id; chapter = 1; $("#bib-book").value = book; }
  else if (by < 0 && i > 0) { book = books[i - 1].id; chapter = books[i - 1].chapters; $("#bib-book").value = book; }
  else return;
  fillChapters();
  show();
}
