/**
 * Pocket TTS Kreyòl on the phone: the fine-tune exported to ONNX (INT8, 512-position cache) and run
 * with onnxruntime-react-native. A port of the reference runtime (KevinAHM/pocket-tts-onnx), checked
 * against PyTorch on 20 held-out sentences before this was written: same CER within run-to-run noise.
 *
 * The loop, per chunk of text: the text conditioner turns token ids into embeddings, which are fed to
 * the LM on top of the voice's keys and values; then, frame by frame (12.5 a second), the LM gives a conditioning
 * vector and an end-of-speech logit, one flow step turns noise into a 32-dim latent, and the Mimi
 * decoder turns latents into 24 kHz audio, a few frames at a time so playback starts early.
 *
 * The LM is the append-only export (export/lomotron/scripts/export_flow_lm_kv.py): it takes each
 * layer's past keys/values at their used length and returns only the new positions. The stock
 * export passed a fixed 100 MB cache in and out every frame, which made a Galaxy A16 ~5x too slow.
 */
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { File } from 'expo-file-system';
import { UnigramTokenizer } from './tokenizer';
import { pocketPrompt, speakable } from './kreyolText';

type Entry = { input_name: string; output_name: string; shape: number[]; dtype: string; fill: string };
type State = Record<string, Tensor>;
type Bundle = {
  latent_dim: number; conditioning_dim: number; sample_rate: number; max_token_per_chunk: number; samples_per_frame: number;
  flow_lm_state_manifest: Entry[]; mimi_state_manifest: Entry[];
  kv: { layers: number; heads: number; head_dim: number; capacity: number };
  mimi_passthrough: string[];   // state outputs that are just their input (Identity): never fetched
  lm_file?: string;             // which LM export to load (INT8 or 4-bit), default the INT8 one
};
/** A voice's keys and values after its audio prompt, per layer. */
type Prefix = { k: Float32Array[]; v: Float32Array[]; length: number };

export const SAMPLE_RATE = 24000;
export const debug = { sink: (_m: string) => {}, stats: '', verbose: false };
const trace = (...a: unknown[]) => { if (debug.verbose) console.log('[pocket]', a.map(String).join(' ')); };
const TEMPERATURE = 0.7;          // the reference runtime's default, which the checks used
const EOS_LOGIT = -4.0;
const TOKENS_PER_SECOND = 3.0;
const PADDING_SECONDS = 2.0;
const FRAME_RATE = 12.5;
const THREADS = 2;                // measured on a Galaxy A16: 2 beat 1, 3 and 4 for the LM and Mimi

function fill(entry: Entry): Tensor {
  const size = entry.shape.reduce((a, b) => a * b, 1);
  if (entry.dtype === 'int64') return new Tensor('int64', new BigInt64Array(size), entry.shape);
  if (entry.dtype === 'bool') return new Tensor('bool', new Uint8Array(size).fill(entry.fill === 'ones' ? 1 : 0), entry.shape);
  const data = new Float32Array(size);
  if (entry.fill === 'nan') data.fill(NaN);
  else if (entry.fill === 'ones') data.fill(1);
  return new Tensor('float32', data, entry.shape);
}

function gaussian(n: number, std: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(Math.random(), 1e-12), v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u)) * std;
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** expo-file-system takes file:// URIs; ONNX Runtime and the rest take plain paths. */
function fileAt(path: string): File {
  return new File(path.startsWith('file://') ? path : `file://${path}`);
}

function readFloats(path: string): Float32Array {
  const bytes = fileAt(path).bytesSync();
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/**
 * Mimi's state outputs, written in place. The binding copies every output it allocates into a new
 * JS buffer; preallocated output tensors (fetches) are written directly instead. Two sets, swapped
 * every run, so a run never writes into its own inputs. Large fixed-shape float entries are
 * preallocated, pass-through entries are not fetched at all, the rest come back new.
 */
class Rotating {
  private sets: Record<string, Tensor>[] = [{}, {}];
  private next = 0;
  private skip: Set<string>;
  constructor(private session: InferenceSession, private manifest: Entry[], passthrough: string[] = []) {
    this.skip = new Set(passthrough);
    for (const e of manifest) {
      if (this.skip.has(e.output_name) || e.dtype !== 'float32' || e.shape.reduce((a, b) => a * b, 1) < 4096) continue;
      for (const set of this.sets) set[e.output_name] = fill(e);
    }
  }
  /** Outputs for the next run: the set not used by the last one. */
  fetches(): Record<string, Tensor | null> {
    const set = this.sets[this.next];
    this.next = 1 - this.next;
    const out: Record<string, Tensor | null> = {};
    for (const name of this.session.outputNames) if (!this.skip.has(name)) out[name] = set[name] ?? null;
    return out;
  }
  /** The state for the next run: the outputs, and the inputs of the entries passed through. */
  next_state(inputs: State, out: InferenceSession.ReturnType): State {
    const state: State = {};
    for (const e of this.manifest) state[e.input_name] = this.skip.has(e.output_name) ? inputs[e.input_name] : out[e.output_name];
    return state;
  }
}

/**
 * Each LM layer's keys and values, time-major so any prefix is contiguous. The binding hands ORT the
 * start of a typed array's ArrayBuffer, so every layer gets its own buffer and views start at 0.
 */
class KVCache {
  private k: ArrayBuffer[] = [];
  private v: ArrayBuffer[] = [];
  length = 0;
  private width: number;
  constructor(private kv: Bundle['kv']) {
    this.width = kv.heads * kv.head_dim;
    for (let i = 0; i < kv.layers; i++) {
      this.k.push(new ArrayBuffer(kv.capacity * this.width * 4));
      this.v.push(new ArrayBuffer(kv.capacity * this.width * 4));
    }
  }
  get room(): number { return this.kv.capacity - this.length; }
  reset(prefix: Prefix): void {
    for (let i = 0; i < this.kv.layers; i++) {
      new Float32Array(this.k[i]).set(prefix.k[i]);
      new Float32Array(this.v[i]).set(prefix.v[i]);
    }
    this.length = prefix.length;
  }
  feeds(): Record<string, Tensor> {
    const dims = [1, this.length, this.kv.heads, this.kv.head_dim];
    const n = this.length * this.width;
    const out: Record<string, Tensor> = {};
    for (let i = 0; i < this.kv.layers; i++) {
      out[`past_k_${i}`] = new Tensor('float32', new Float32Array(this.k[i], 0, n), dims);
      out[`past_v_${i}`] = new Tensor('float32', new Float32Array(this.v[i], 0, n), dims);
    }
    return out;
  }
  append(out: InferenceSession.ReturnType, positions: number): void {
    if (positions > this.room) throw new Error('KV cache full');
    for (let i = 0; i < this.kv.layers; i++) {
      new Float32Array(this.k[i]).set(out[`new_k_${i}`].data as Float32Array, this.length * this.width);
      new Float32Array(this.v[i]).set(out[`new_v_${i}`].data as Float32Array, this.length * this.width);
    }
    this.length += positions;
  }
}

export class Pocket {
  private bundle!: Bundle;
  private tokenizer!: UnigramTokenizer;
  private conditioner!: InferenceSession;
  private main!: InferenceSession;
  private flow!: InferenceSession;
  private decoder!: InferenceSession;
  private bos!: Float32Array;
  private voices = new Map<string, Prefix>();
  private cache!: KVCache;
  private mimiOut!: Rotating;

  constructor(private dir: string) {}

  async load(): Promise<void> {
    // Few threads: the frame loop is memory-bound, and on big.LITTLE phones 8 threads were slowest.
    const opts: InferenceSession.SessionOptions = { executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: THREADS };
    this.bundle = JSON.parse(fileAt(`${this.dir}/bundle.json`).textSync());
    this.tokenizer = new UnigramTokenizer(JSON.parse(fileAt(`${this.dir}/tokenizer.json`).textSync()));
    this.bos = readFloats(`${this.dir}/bos_before_voice.f32`);
    [this.conditioner, this.main, this.flow, this.decoder] = await Promise.all([
      InferenceSession.create(`${this.dir}/text_conditioner_int8.onnx`, opts),
      InferenceSession.create(`${this.dir}/${this.bundle.lm_file ?? 'flow_lm_main_kv_int8.onnx'}`, opts),
      InferenceSession.create(`${this.dir}/flow_lm_flow_int8.onnx`, opts),
      InferenceSession.create(`${this.dir}/mimi_decoder_int8.onnx`, opts),
    ]);
    this.cache = new KVCache(this.bundle.kv);
    this.mimiOut = new Rotating(this.decoder, this.bundle.mimi_state_manifest, this.bundle.mimi_passthrough ?? []);
  }

  /** Dev check: LM ms per frame (voice + ~100 past) by ORT threads, run alone. */
  async bench(voice: string): Promise<string> {
    const { latent_dim: L, conditioning_dim: D } = this.bundle;
    const base = await this.voiceState(voice);
    const lines: string[] = [];
    for (const threads of [1, 2, 3, 4]) {
      const session = await InferenceSession.create(`${this.dir}/${this.bundle.lm_file ?? 'flow_lm_main_kv_int8.onnx'}`,
        { executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: threads });
      this.cache.reset(base);
      const times: number[] = [];
      for (let i = 0; i < 40; i++) {
        const t0 = Date.now();
        const out = await session.run({
          sequence: new Tensor('float32', new Float32Array(L).fill(i ? 0.1 : NaN), [1, 1, L]),
          text_embeddings: new Tensor('float32', new Float32Array(0), [1, 0, D]),
          ...this.cache.feeds(),
        });
        this.cache.append(out, 1);
        times.push(Date.now() - t0);
      }
      await session.release();
      lines.push(`LM ${threads} thr: ${times.slice(5).sort((x, y) => x - y)[17]} ms`);
      debug.sink(lines.join(' | '));
    }
    return lines.join('\n');
  }

  private initState(manifest: Entry[]): State {
    const state: State = {};
    for (const e of manifest) state[e.input_name] = fill(e);
    return state;
  }

  private nextState(manifest: Entry[], out: InferenceSession.ReturnType): State {
    const state: State = {};
    for (const e of manifest) state[e.input_name] = out[e.output_name];
    return state;
  }

  /** The LM's keys and values after hearing the voice's clip, made once per voice. */
  async voiceState(name: string): Promise<Prefix> {
    const cached = this.voices.get(name);
    if (cached) return cached;
    const meta = JSON.parse(fileAt(`${this.dir}/voices/voices.json`).textSync())[name];
    const emb = readFloats(`${this.dir}/voices/${meta.file}`);
    const dim = this.bundle.conditioning_dim;
    const frames = emb.length / dim;
    const prompt = new Float32Array((frames + 1) * dim);
    prompt.set(this.bos, 0);                       // the fine-tune inserts BOS before the voice
    prompt.set(emb, dim);
    this.cache.length = 0;
    const out = await this.main.run({
      sequence: new Tensor('float32', new Float32Array(0), [1, 0, this.bundle.latent_dim]),
      text_embeddings: new Tensor('float32', prompt, [1, frames + 1, dim]),
      ...this.cache.feeds(),
    });
    const prefix: Prefix = { k: [], v: [], length: frames + 1 };
    for (let i = 0; i < this.bundle.kv.layers; i++) {
      prefix.k.push(out[`new_k_${i}`].data as Float32Array);
      prefix.v.push(out[`new_v_${i}`].data as Float32Array);
    }
    this.voices.set(name, prefix);
    return prefix;
  }

  /** Text -> chunks of at most max_token_per_chunk tokens, split at sentence ends, then commas. */
  chunks(text: string): string[] {
    const tok = this.tokenizer;
    const limit = this.bundle.max_token_per_chunk;
    const boundaries = (ids: number[], marks: Set<number>) => {
      const at = [0];
      let prev = false;
      ids.forEach((id, i) => {
        if (marks.has(id)) prev = true;
        else { if (prev) at.push(i); prev = false; }
      });
      at.push(ids.length);
      return at;
    };
    const segments = (ids: number[], at: number[]) =>
      at.slice(0, -1).map((s, i) => ({ count: at[i + 1] - s, text: tok.decode(ids.slice(s, at[i + 1])) }));
    const ids = tok.encode(pocketPrompt(text).text.trim());
    const ends = new Set(tok.encode('.!...?').slice(1));
    const commas = new Set(tok.encode(',;:').slice(1));
    const refined: { count: number; text: string }[] = [];
    for (const seg of segments(ids, boundaries(ids, ends))) {
      if (seg.count <= limit) { refined.push(seg); continue; }
      const sub = tok.encode(seg.text.trim());
      const parts = segments(sub, boundaries(sub, commas));
      refined.push(...(parts.length > 1 ? parts : [seg]));
    }
    const out: string[] = [];
    let current = '', count = 0;
    for (const seg of refined) {
      if (!current) { current = seg.text; count = seg.count; continue; }
      if (count + seg.count > limit) { out.push(current.trim()); current = seg.text; count = seg.count; }
      else { current += ' ' + seg.text; count += seg.count; }
    }
    if (current) out.push(current.trim());
    return out;
  }

  /**
   * Speak `text` with `voice`, handing audio (Float32, 24 kHz) to onAudio as soon as it is decoded:
   * the first 2 frames (~160 ms) right away, then 3 at a time while the buffer is short, else 15.
   */
  async speak(text: string, voice: string, onAudio: (pcm: Float32Array) => void,
              stopped: () => boolean = () => false): Promise<void> {
    const { latent_dim: L, conditioning_dim: D, mimi_state_manifest: mimiManifest } = this.bundle;
    trace('speak: voice state');
    const base = await this.voiceState(voice);
    const t = { main: 0, flow: 0, dec: 0, cond: 0, frames: 0, decodes: 0 };
    const notes: string[] = [];
    const clock = () => Date.now();
    const chunks = this.chunks(speakable(text)).map((chunk) => {
      const { text: prepared, framesAfterEos } = pocketPrompt(chunk);
      return { prepared, framesAfterEos, ids: this.tokenizer.encode(prepared) };
    });
    trace('speak: chunks', chunks.length);

    // Each sentence chunk starts playing once the audio in hand outlasts the rest of that chunk's
    // generation, at the rate measured so far: a phone slower than real time plays each sentence
    // without gaps and waits, if it must, between sentences, where a pause sounds natural. A phone
    // faster than real time starts after 2 frames. Frames per chunk are estimated from its tokens.
    const estimate = (tokens: number) => 2.3 * tokens + 10;   // high end of 13-43 token chunks seen
    let expected = 0, generated = 0, playing = false, heldFrames = 0;
    let playStart = 0, given = 0, gaps = 0;        // seconds of audio handed to the speaker; underruns
    let decodedFrames = 0;
    const starts: string[] = [];
    const held: Float32Array[] = [];
    const give = (pcm: Float32Array) => {
      if (playStart && (clock() - playStart) / 1000 > given + 0.05) { gaps++; playStart = clock() - given * 1000; }
      given += pcm.length / SAMPLE_RATE;
      onAudio(pcm);
    };
    const flush = () => {
      // Resuming after a hold: if the speaker ran dry meanwhile, that was a pause between
      // sentences, not a gap, so the timeline restarts.
      if (!playing && (!playStart || (clock() - playStart) / 1000 > given)) playStart = clock() - given * 1000;
      playing = true;
      held.splice(0).forEach(give);
    };
    const emit = (pcm: Float32Array) => {
      if (playing) { give(pcm); return; }
      held.push(pcm);
      heldFrames += pcm.length / this.bundle.samples_per_frame;
      // Cost per frame of each stage (Mimi's mostly batched), +15% for the slowdown as the cache
      // grows and the phone warms.
      const perFrame = 1.15 * ((t.main + t.flow) / Math.max(1, t.frames) + t.dec / Math.max(1, decodedFrames)) + 3;
      const rate = 1000 / perFrame;
      const remaining = Math.max(0, expected - generated);
      const needed = rate >= FRAME_RATE ? 2 : Math.max(2, remaining * (FRAME_RATE / rate - 1));
      if (heldFrames >= needed) {
        starts.push(`${Math.round(heldFrames)}f@${rate.toFixed(1)}f/s`);
        flush();
      }
    };
    /** A new chunk: hold its audio until it can play through; what the speaker still has counts. */
    const regate = () => {
      if (!playing) return;
      playing = false;
      heldFrames = Math.max(0, given - (clock() - playStart) / 1000) * FRAME_RATE;
    };

    for (const { prepared, framesAfterEos, ids } of chunks) {
      if (stopped()) return;
      expected = generated + estimate(ids.length);
      regate();
      trace('chunk', JSON.stringify(prepared), ids.length, 'tokens');
      let c0 = clock();
      const embedded = await this.conditioner.run({
        token_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      });
      const textEmb = embedded[this.conditioner.outputNames[0]];
      t.cond += clock() - c0;
      c0 = clock();
      this.cache.reset(base);
      let out = await this.main.run({
        sequence: new Tensor('float32', new Float32Array(0), [1, 0, L]),
        text_embeddings: textEmb.dims.length === 2 ? new Tensor('float32', textEmb.data as Float32Array, [1, ...textEmb.dims]) : textEmb,
        ...this.cache.feeds(),
      });
      this.cache.append(out, ids.length);
      trace('text prompted');
      let mimi = this.initState(mimiManifest);
      const pending: Float32Array[] = [];
      // Decodes run between LM frames, not alongside: run in parallel, the two fought over memory
      // bandwidth and the LM frame doubled on a Galaxy A16. The queue keeps them in order.
      let decoding: Promise<void> = Promise.resolve();
      const decode = (n: number) => {
        const take = pending.splice(0, n);
        const latents = new Float32Array(take.length * L);
        take.forEach((f, i) => latents.set(f, i * L));
        decoding = decoding.then(async () => {
          if (stopped()) return;
          const d0 = clock();
          const res = await this.decoder.run({ latent: new Tensor('float32', latents, [1, take.length, L]), ...mimi }, this.mimiOut.fetches());
          t.dec += clock() - d0;
          t.decodes++;
          decodedFrames += take.length;
          mimi = this.mimiOut.next_state(mimi, res);
          emit(res[this.decoder.outputNames[0]].data as Float32Array);
        });
        return take.length;
      };
      let curr: Float32Array = new Float32Array(L).fill(NaN);
      let eosStep = -1, decoded = 0;
      const limit = Math.min(Math.ceil((ids.length / TOKENS_PER_SECOND + PADDING_SECONDS) * 12.5), this.cache.room);
      t.cond += clock() - c0;
      let step = 0;
      for (; step < limit && !stopped(); step++) {
        const m0 = clock();
        out = await this.main.run({
          sequence: new Tensor('float32', curr, [1, 1, L]),
          text_embeddings: new Tensor('float32', new Float32Array(0), [1, 0, D]),
          ...this.cache.feeds(),
        });
        this.cache.append(out, 1);
        t.main += clock() - m0;
        t.frames++;
        generated++;
        if ((out.eos_logit.data as Float32Array)[0] > EOS_LOGIT && eosStep < 0) eosStep = step;
        if (eosStep >= 0 && step >= eosStep + framesAfterEos) break;
        let x: Float32Array = gaussian(L, Math.sqrt(TEMPERATURE));
        const f0 = clock();
        const flow = await this.flow.run({
          c: out.conditioning,
          s: new Tensor('float32', new Float32Array([0]), [1, 1]),
          t: new Tensor('float32', new Float32Array([1]), [1, 1]),
          x: new Tensor('float32', x, [1, L]),
        });
        const v = flow[this.flow.outputNames[0]].data as Float32Array;
        t.flow += clock() - f0;
        x = x.map((xi, i) => xi + v[i]);           // one LSD step: dt = 1
        pending.push(x);
        curr = x;
        // The first 2 frames are decoded at once (playback may start on them); afterwards decode in
        // batches (Mimi's cost per call is partly fixed), smaller when the speaker's buffer runs low.
        if (!decoded && pending.length >= 2) decoded += decode(2);
        else if (decoded) {
          const bufferSec = playing ? given - (clock() - playStart) / 1000 : Infinity;
          // 8 frames take ~0.8 s to make on a slow phone: decode sooner when the buffer is shorter.
          if (pending.length >= 8 || (pending.length >= 3 && bufferSec < 1.2)) decoded += decode(pending.length);
        }
        await decoding;
      }
      if (pending.length && !stopped()) decode(pending.length);
      await decoding;
      notes.push(`${ids.length}tok→${step}f eos@${eosStep}/${limit}`);
    }
    if (!stopped()) flush();
    const per = (ms: number) => (t.frames ? (ms / t.frames).toFixed(0) : '-');
    debug.stats = `per frame: LM ${per(t.main)} ms, flow ${per(t.flow)} ms, Mimi ${per(t.dec)} ms (${t.decodes} calls); starts ${starts.join(', ')}; gaps ${gaps}; prompt ${t.cond} ms | ${notes.join(' · ')}`;
  }
}
