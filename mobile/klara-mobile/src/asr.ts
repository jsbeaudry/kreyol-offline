/**
 * Kreyòl speech to text on the phone: Qwen3-ASR 0.6B v2 (GGUF Q4_K_M + its Q8_0 projector) run by
 * llama.rn, asked exactly what llama-server asks it — "Transcribe audio to text" with the audio after.
 */
import { initLlama, type LlamaContext } from 'llama.rn';

export const ASR_MODEL = 'qwen3-asr-0.6b-kreyol-v2-Q4_K_M.gguf';
export const ASR_PROJECTOR = 'mmproj-qwen3-asr-0.6b-kreyol-v2-Q8_0.gguf';

// What the fine-tune writes in front of every transcript.
const TAG = /^\s*language\s+[^<]{0,40}<asr_text>/i;
// What it writes when the clip held no speech: it echoes its instruction (seen 16 times in one noisy
// session on the desktop), or loops on one phrase.
const NOT_SPEECH = /^(transcri\w*|transkri\w*|audio|odyo)$/;

export function isNoise(text: string): boolean {
  const words = (text.toLowerCase().match(/[\p{L}]+/gu) ?? []);
  if (!words.length || NOT_SPEECH.test(words.join(' '))) return true;
  return words.length >= 8 && new Set(words).size / words.length < 0.3;
}

export class Ears {
  private ctx: LlamaContext | null = null;

  async load(dir: string, threads?: number): Promise<void> {
    this.ctx = await initLlama({
      model: `${dir}/${ASR_MODEL}`,
      n_ctx: 2048,
      ...(threads ? { n_threads: threads } : {}),
      n_gpu_layers: 0,          // CPU: a 0.6B model, and Android GPU backends vary by phone
      ctx_shift: false,         // required for multimodal
    });
    const ok = await this.ctx.initMultimodal({ path: `${dir}/${ASR_PROJECTOR}`, use_gpu: false });
    if (!ok) throw new Error('the audio projector did not load');
  }

  /** A 16 kHz mono WAV file -> Kreyòl text, or '' when it heard nothing but noise. */
  async transcribe(wavPath: string): Promise<string> {
    const text = (await this.run(wavPath)).replace(TAG, '').split(/\s+/).join(' ').trim();
    return isNoise(text) ? '' : text;
  }

  private async run(wavPath: string): Promise<string> {
    if (!this.ctx) throw new Error('ASR not loaded');
    const result = await this.ctx.completion({
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'Transcribe audio to text' },
          { type: 'input_audio', input_audio: { url: wavPath.startsWith('file://') ? wavPath : `file://${wavPath}`, format: 'wav' } },
        ],
      }],
      n_predict: 192,
      temperature: 0,
    });
    return result.text || '';
  }

  async release(): Promise<void> {
    await this.ctx?.release();
    this.ctx = null;
  }
}
