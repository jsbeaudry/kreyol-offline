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
    makeBlocks();
    $("#bib-lines").replaceChildren(...lines.map((text, i) => el("div", { class: "bib-line", "data-i": String(i) },
      el("button", { type: "button", class: "n", title: "Tande depi liy sa a (hear it from this line)", onclick: () => read(i) }, String(i + 1)),
      el("p", {}, text))));
    $("#bib-meta").textContent = `${c.name} ${c.chapter} · ${lines.length} liy · ${c.chars.toLocaleString("fr")} karaktè · ` +
      `anviwon ${length(c.chars / 10)} odyo (about ${length(c.chars / 10)} to listen to)`;
    say(status(), "");
    remember();
  } catch (e) { fail(status(), e); }
}

/* ---------- reading aloud ----------

   Lines are read in blocks of a few at a time: fewer starts, and the voice keeps its rhythm across a
   sentence boundary. The server has one voice model, so two blocks are never made at the same time; what
   runs in parallel is making the next blocks while the current one plays. Measured on this Mac, making
   audio takes about 0.79 s per second of speech, so each block buys time for the next one.

   A block is only ready in time if it is no longer than about a quarter more than the one playing, so
   blocks are cut by characters rather than by line count: three lines of Matye 5 can be 46 characters or
   287. Simulated over six chapters, blocks of about 110 characters never leave a silence, and the voice
   starts about 11 s after the button; smaller blocks start sooner but run dry in the middle, which is
   worse to listen to.
*/

const BLOCK_CHARS = 110, MAX_BLOCK_LINES = 4, MAX_BLOCK_CHARS = 260, AHEAD = 3;
let blocks = [];
let queue = Promise.resolve();
let epoch = 0;                     // stopping, or changing chapter or voice, abandons what was queued

function makeBlocks() {
  blocks = [];
  let current = null;
  lines.forEach((text, i) => {
    const full = current && (current.chars >= BLOCK_CHARS || current.lines.length >= MAX_BLOCK_LINES ||
                             current.chars + text.length > MAX_BLOCK_CHARS);
    if (!current || full) {
      current = { lines: [], chars: 0 };
      blocks.push(current);
    }
    current.lines.push(i);
    current.chars += text.length + 1;
  });
}
const blockOf = (line) => Math.max(0, blocks.findIndex((b) => b.lines.includes(line)));

function trimCache() {
  while (cache.size > KEEP) {
    const [k, held] = cache.entries().next().value;
    URL.revokeObjectURL(held.url);
    cache.delete(k);
  }
}
const key = (bi) => `${picker.value()}|${book}|${chapter}|${bi}`;

async function prepare(bi) {
  if (bi < 0 || bi >= blocks.length) return null;
  const k = key(bi);
  if (!cache.has(k)) {
    const text = blocks[bi].lines.map((i) => lines[i]).join(" ");
    const j = await post("/api/speak", { text, voice: picker.value() }, true);
    const blob = new Blob([Uint8Array.from(atob(j.audio), (c) => c.charCodeAt(0))], { type: "audio/wav" });
    cache.set(k, { url: URL.createObjectURL(blob), seconds: j.audio_seconds });
    trimCache();
  }
  return cache.get(k);
}

function queueAhead(bi) {                 // keep the next blocks coming, one at a time, quietly
  const mine = epoch;
  for (let k = 1; k <= AHEAD; k++) {
    const next = bi + k;
    if (next < blocks.length && !cache.has(key(next))) {
      queue = queue.then(() => (epoch === mine && !cache.has(key(next)) ? prepare(next) : null)).catch(() => {});
    }
  }
}

function mark(i) {
  if (i === current) return;
  document.querySelectorAll("#bib-lines .bib-line.playing").forEach((row) => row.classList.remove("playing"));
  current = i;
  const row = document.querySelector(`#bib-lines .bib-line[data-i="${i}"]`);
  if (row) { row.classList.add("playing"); row.scrollIntoView({ block: "nearest" }); }
}

function playBlock(bi, held) {
  const block = blocks[bi];
  const total = block.lines.reduce((sum, i) => sum + lines[i].length, 0) || 1;
  // no word times come back, so within a block the line is followed by its share of the characters
  const follow = () => {
    const length = audio.duration || held.seconds || 0;
    let spent = 0;
    for (const i of block.lines) {
      spent += length * lines[i].length / total;
      if (audio.currentTime < spent) { mark(i); return; }
    }
    mark(block.lines[block.lines.length - 1]);
  };
  return new Promise((resolve) => {
    audio.src = held.url;
    audio.ontimeupdate = follow;
    audio.onended = audio.onerror = () => { audio.ontimeupdate = null; resolve(); };
    mark(block.lines[0]);
    audio.play().catch(() => { audio.ontimeupdate = null; resolve(); });
  });
}

async function read(fromLine) {
  if (playing) stop();
  playing = true;
  $("#bib-stop").hidden = false;
  $("#bib-read").disabled = true;
  for (let bi = blockOf(fromLine); bi < blocks.length && playing; bi++) {
    queueAhead(bi);
    let held = cache.get(key(bi));
    if (!held) {
      say(status(), "Ap prepare vwa a… (getting the voice ready…)");
      try { held = await prepare(bi); } catch (e) { fail(status(), e); break; }
    }
    if (!playing) break;
    const block = blocks[bi];
    const first = block.lines[0] + 1, last = block.lines[block.lines.length - 1] + 1;
    say(status(), `Ap li liy ${first === last ? first : `${first}-${last}`} nan ${lines.length}. ` +
                  `(Reading line${first === last ? "" : "s"} ${first === last ? first : `${first}-${last}`} of ${lines.length}.)`);
    await playBlock(bi, held);
  }
  if (playing && current >= lines.length - 1) say(status(), "Fini chapit la. (End of the chapter.)");
  stop(true);
}

function stop(keepMessage) {
  playing = false;
  epoch += 1;
  audio.pause();
  audio.ontimeupdate = null;
  $("#bib-stop").hidden = true;
  $("#bib-read").disabled = false;
  document.querySelectorAll("#bib-lines .bib-line.playing").forEach((row) => row.classList.remove("playing"));
  current = -1;
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
  picker = voicePicker($('[data-voices="bib"]'), { onChange: () => { if (!playing) say(status(), ""); } });
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
