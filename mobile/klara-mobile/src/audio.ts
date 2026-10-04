/**
 * Microphone in, voice out, with react-native-audio-api: the recorder hands over 16 kHz Float32
 * buffers that are kept and written as a WAV for the ASR; Pocket's 24 kHz chunks are queued for
 * playback as they arrive, so speech starts with the first decoded frames.
 */
import { AudioContext, AudioManager, AudioRecorder, type AudioBufferQueueSourceNode } from 'react-native-audio-api';
import { File, Paths } from 'expo-file-system';

export const MIC_RATE = 16000;

export async function micAllowed(): Promise<boolean> {
  return (await AudioManager.requestRecordingPermissions()) === 'Granted';
}

export function wavBytes(pcm: Float32Array, rate: number): Uint8Array {
  const out = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(out);
  const text = (at: number, s: string) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); text(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  text(36, 'data'); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 32767, true);
  return new Uint8Array(out);
}

export class Mic {
  private recorder = new AudioRecorder();
  private parts: Float32Array[] = [];

  async start(): Promise<void> {
    this.parts = [];
    this.recorder.onAudioReady({ sampleRate: MIC_RATE, bufferLength: 1600, channelCount: 1 }, (event) => {
      this.parts.push(new Float32Array(event.buffer.getChannelData(0)));
    });
    await this.recorder.start();
  }

  /** Stop, and write what was heard as a WAV. Returns its path and length in seconds. */
  async stop(): Promise<{ path: string; seconds: number }> {
    await this.recorder.stop();
    this.recorder.clearOnAudioReady();
    const pcm = new Float32Array(this.parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of this.parts) { pcm.set(p, at); at += p.length; }
    const file = new File(Paths.cache, 'heard.wav');
    if (file.exists) file.delete();
    file.create();
    file.write(wavBytes(pcm, MIC_RATE));
    return { path: file.uri, seconds: pcm.length / MIC_RATE };
  }
}

export class Speaker {
  private ctx: AudioContext | null = null;
  private queue: AudioBufferQueueSourceNode | null = null;

  begin(rate: number): void {
    this.stop();
    this.ctx = new AudioContext({ sampleRate: rate });
    this.queue = this.ctx.createBufferQueueSource();
    this.queue.connect(this.ctx.destination);
    this.queue.start(this.ctx.currentTime, 0);   // its default offset (-1) is rejected by its own check
  }

  push(pcm: Float32Array): void {
    if (!this.ctx || !this.queue || !pcm.length) return;
    const buffer = this.ctx.createBuffer(1, pcm.length, this.ctx.sampleRate);
    buffer.copyToChannel(new Float32Array(pcm), 0);
    this.queue.enqueueBuffer(buffer);
  }

  stop(): void {
    try { this.queue?.stop(); } catch { /* already stopped */ }
    this.ctx?.close();
    this.ctx = null;
    this.queue = null;
  }
}
