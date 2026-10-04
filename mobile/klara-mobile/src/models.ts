/**
 * Where the models live on the phone — the app's documents folder — and fetching them on first launch.
 *
 * DOWNLOADS points at two public Hugging Face repos (made public 2026-10-04), read without a token:
 * ~893 MB in 16 files, about 2 minutes on home Wi-Fi. Development builds can still get the models
 * by `adb` instead (push-models.sh).
 */
import { Platform } from 'react-native';
import { Directory, File, Paths } from 'expo-file-system';
import { ASR_MODEL, ASR_PROJECTOR } from './asr';

// Downloads land in the app's private documents. On Android, models copied with adb into the app's
// external folder are used first — a release build is not debuggable, so adb cannot reach the other.
const DOWNLOADED = new Directory(Paths.document, 'models');
const SIDELOADED = Platform.OS === 'android'
  ? new Directory('file:///storage/emulated/0/Android/data/com.jsbeaudry.klaramobile/files/models') : null;

function root(): Directory {
  if (SIDELOADED && new File(SIDELOADED, 'asr', ASR_MODEL).exists) return SIDELOADED;
  return DOWNLOADED;
}
export const MODELS = root();
export const POCKET_DIR = new Directory(MODELS, 'pocket');
export const ASR_DIR = new Directory(MODELS, 'asr');

const POCKET_FILES = ['bundle.json', 'tokenizer.json', 'bos_before_voice.f32', 'text_conditioner_int8.onnx',
  'flow_lm_main_kv_q4.onnx', 'flow_lm_flow_int8.onnx', 'mimi_decoder_int8.onnx', 'voices/voices.json',
  ...['reel', 'vocal', 'female1', 'female2', 'man1', 'man2'].map((v) => `voices/${v}.f32`)];

const HF = 'https://huggingface.co';
export const DOWNLOADS: { dir: Directory; name: string; url: string }[] = [
  { dir: ASR_DIR, name: ASR_MODEL, url: `${HF}/jsbeaudry/qwen3-asr-0.6b-kreyol-v2-GGUF/resolve/main/${ASR_MODEL}` },
  { dir: ASR_DIR, name: ASR_PROJECTOR, url: `${HF}/jsbeaudry/qwen3-asr-0.6b-kreyol-v2-GGUF/resolve/main/${ASR_PROJECTOR}` },
  ...POCKET_FILES.map((name) => ({
    dir: POCKET_DIR, name, url: `${HF}/jsbeaudry/pocket-tts-kreyol-onnx/resolve/main/${name}`,
  })),
];

export function path(dir: Directory, name = ''): string {
  return decodeURI(new File(dir, name).uri).replace(/^file:\/\//, '');
}

export function missing(): string[] {
  return DOWNLOADS.filter((d) => !new File(d.dir, d.name).exists).map((d) => d.name);
}

/** Download whatever is missing, one file at a time, reporting a 0-1 fraction of files done. */
export async function fetchMissing(onProgress: (done: number, current: string) => void): Promise<void> {
  const todo = DOWNLOADS.filter((d) => !new File(d.dir, d.name).exists);
  let done = 0;
  for (const d of todo) {
    onProgress(done / todo.length, d.name);
    const target = new File(d.dir, d.name);
    const parent = new Directory(target.parentDirectory.uri);
    if (!parent.exists) parent.create({ intermediates: true });
    await File.downloadFileAsync(d.url, target);
    done += 1;
  }
  onProgress(1, '');
}
