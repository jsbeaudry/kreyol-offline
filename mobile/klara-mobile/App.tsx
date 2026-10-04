/**
 * Klara mobile — a speech demo, fully on the phone:
 *   Tande  hold the button and speak; Qwen3-ASR 0.6B Kreyòl writes what it heard.
 *   Pale   type Kreyòl; Pocket TTS Kreyòl reads it in one of six voices, streaming.
 */
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Ears } from './src/asr';
import { Mic, Speaker, micAllowed, wavBytes } from './src/audio';
import { File, Paths } from 'expo-file-system';
import { ASR_DIR, POCKET_DIR, fetchMissing, missing, path } from './src/models';
import { Pocket, SAMPLE_RATE, debug } from './src/pocket';

const VOICES = [
  { id: 'reel', label: 'Reel', note: 'fanm' }, { id: 'female1', label: 'Fanm 1', note: 'pi klè' },
  { id: 'female2', label: 'Fanm 2', note: 'fanm' }, { id: 'vocal', label: 'Vocal', note: 'fanm' },
  { id: 'man1', label: 'Gason 1', note: 'gason' }, { id: 'man2', label: 'Gason 2', note: 'gason' },
];
const SAMPLE = 'Bonjou! Mwen se Klara. Ayiti te pran endepandans li nan 1804, sou direksyon Jean-Jacques Dessalines.';

type Phase = 'checking' | 'download' | 'downloading' | 'loading' | 'ready' | 'error';

export default function App() {
  const [phase, setPhase] = useState<Phase>('checking');
  const [detail, setDetail] = useState('');
  const [progress, setProgress] = useState(0);
  const ears = useRef(new Ears());
  const pocket = useRef<Pocket | null>(null);
  const mic = useRef<Mic>(null as unknown as Mic);
  const speaker = useRef<Speaker>(null as unknown as Speaker);
  if (!mic.current) mic.current = new Mic();
  if (!speaker.current) speaker.current = new Speaker();

  // Tande
  const [listening, setListening] = useState(false);
  const [hearing, setHearing] = useState(false);
  const [heard, setHeard] = useState('');
  const [heardInfo, setHeardInfo] = useState('');

  // Pale
  const [text, setText] = useState(SAMPLE);
  const [voice, setVoice] = useState('reel');
  const [speaking, setSpeaking] = useState(false);
  const [spokeInfo, setSpokeInfo] = useState('');
  const stopFlag = useRef(false);

  async function loadModels() {
    setPhase('loading');
    try {
      const t0 = Date.now();
      setDetail('Zòrèy (Qwen3-ASR)…');
      await ears.current.load(path(ASR_DIR));
      setDetail('Vwa (Pocket TTS)…');
      const p = new Pocket(path(POCKET_DIR));
      await p.load();
      await p.voiceState(voice);
      pocket.current = p;
      setDetail(`Pare an ${((Date.now() - t0) / 1000).toFixed(1)} s`);
      setPhase('ready');
    } catch (error) {
      setDetail(String(error));
      setPhase('error');
    }
  }

  useEffect(() => {
    const gone = missing();
    if (gone.length) {
      setDetail(`${gone.length} fichye pou telechaje (~900 MB)`);
      setPhase('download');
    } else {
      loadModels();
    }
    return () => { speaker.current.stop(); ears.current.release(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function download() {
    setPhase('downloading');
    try {
      await fetchMissing((done, current) => { setProgress(done); setDetail(current); });
      await loadModels();
    } catch (error) {
      setDetail(`Telechajman an echwe: ${error}`);
      setPhase('download');
    }
  }

  async function startListening() {
    if (listening || hearing) return;
    if (!(await micAllowed())) { setHeardInfo('Bezwen pèmisyon mikwofòn nan.'); return; }
    speaker.current.stop();
    await mic.current.start();
    setListening(true);
    setHeardInfo('Ap koute…');
  }

  async function stopListening() {
    if (!listening) return;
    setListening(false);
    setHearing(true);
    try {
      const { path: wav, seconds } = await mic.current.stop();
      if (seconds < 0.3) { setHeardInfo('Twò kout. Kenbe bouton an pandan w ap pale.'); return; }
      const t0 = Date.now();
      const said = await ears.current.transcribe(wav);
      const took = (Date.now() - t0) / 1000;
      setHeard(said || '(anyen — sanble se bri sèlman)');
      setHeardInfo(`${seconds.toFixed(1)} s odyo · transkri an ${took.toFixed(2)} s`);
    } catch (error) {
      setHeardInfo(`Erè: ${error}`);
    } finally {
      setHearing(false);
    }
  }

  // Developer check, for a device that cannot be spoken to (an emulator): long-press "Tande" to
  // transcribe cache/test-heard.wav, if one was copied there with adb.
  async function transcribeTestFile() {
    const test = new File(Paths.cache, 'test-heard.wav');
    if (!test.exists || phase !== 'ready') return;
    setHearing(true);
    try {
      const t0 = Date.now();
      const said = await ears.current.transcribe(test.uri);
      setHeard(said || '(anyen)');
      setHeardInfo(`fichye tès · transkri an ${((Date.now() - t0) / 1000).toFixed(2)} s`);
    } catch (error) {
      setHeardInfo(`Erè: ${error}`);
    } finally {
      setHearing(false);
    }
  }

  // Developer check: long-press "Pale" to time one LM frame per thread count.
  async function benchVoice() {
    const p = pocket.current;
    if (!p || speaking) return;
    setSpeaking(true);
    debug.sink = (m) => setSpokeInfo(`… ${m}`);
    try { setSpokeInfo(await p.bench(voice)); } catch (error) { setSpokeInfo(`Erè: ${error}`); } finally { setSpeaking(false); }
  }

  // Developer check: long-press the title to time a fresh transcription of cache/test-heard.wav
  // per llama.rn thread count (a fresh context each, so nothing is cached).
  async function benchEars() {
    const test = new File(Paths.cache, 'test-heard.wav');
    if (!test.exists || phase !== 'ready') return;
    setHearing(true);
    const lines: string[] = [];
    try {
      for (const threads of [2, 4, 6, 8]) {
        const e = new Ears();
        await e.load(path(ASR_DIR), threads);
        const t0 = Date.now();
        const said = await e.transcribe(test.uri);
        lines.push(`${threads} thr: ${((Date.now() - t0) / 1000).toFixed(2)} s ${said.slice(0, 20)}`);
        await e.release();
        setHeardInfo(lines.join(' | '));
      }
    } catch (error) {
      lines.push(`Erè: ${error}`);
    } finally {
      setHeardInfo(`ASR bench: ${lines.join(' | ')}`);
      setHearing(false);
    }
  }

  async function speak() {
    const p = pocket.current;
    if (!p) return;
    if (speaking) { stopFlag.current = true; speaker.current.stop(); return; }
    stopFlag.current = false;
    setSpeaking(true);
    setSpokeInfo('');
    debug.sink = (m) => setSpokeInfo(`… ${m}`);
    const t0 = Date.now();
    let first = -1;
    let samples = 0;
    try {
      speaker.current.begin(SAMPLE_RATE);
      const parts: Float32Array[] = [];
      await p.speak(text, voice, (pcm) => {
        if (first < 0) first = Date.now() - t0;
        samples += pcm.length;
        parts.push(pcm);
        speaker.current.push(pcm);
      }, () => stopFlag.current);
      // The last thing said, kept as a WAV — to share it, or to check it off the phone.
      const all = new Float32Array(samples);
      parts.reduce((at, part) => { all.set(part, at); return at + part.length; }, 0);
      const last = new File(Paths.cache, 'last-spoken.wav');
      if (last.exists) last.delete();
      last.create();
      last.write(wavBytes(all, SAMPLE_RATE));
      const work = (Date.now() - t0) / 1000;
      const audio = samples / SAMPLE_RATE;
      setSpokeInfo(`${debug.stats}\npremye son ${first} ms · ${audio.toFixed(1)} s odyo an ${work.toFixed(1)} s (RTF ${(work / Math.max(audio, 0.01)).toFixed(2)})`);
    } catch (error) {
      setSpokeInfo(`Erè: ${error instanceof Error ? `${error.message}\n${(error.stack || '').split('\n').slice(0, 4).join('\n')}` : error}`);
    } finally {
      setSpeaking(false);
    }
  }

  return (
    <View style={s.root}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={s.page} keyboardShouldPersistTaps="handled">
        <Text style={s.title} onLongPress={benchEars}>Klara</Text>
        <Text style={s.sub}>Kreyòl sou telefòn nan — san entènèt</Text>

        {phase !== 'ready' && (
          <View style={s.card}>
            {phase === 'download' && (
              <>
                <Text style={s.body}>Modèl yo poko sou telefòn nan. {detail}</Text>
                <Pressable style={s.primary} onPress={download}><Text style={s.primaryText}>Telechaje modèl yo</Text></Pressable>
              </>
            )}
            {(phase === 'downloading' || phase === 'loading' || phase === 'checking') && (
              <View style={s.row}>
                <ActivityIndicator color="#4aa3ff" />
                <Text style={s.body}>
                  {phase === 'downloading' ? `Ap telechaje ${Math.round(progress * 100)}% — ${detail}` : `Ap chaje ${detail}`}
                </Text>
              </View>
            )}
            {phase === 'error' && <Text style={s.error}>{detail}</Text>}
          </View>
        )}

        <View style={[s.card, phase !== 'ready' && s.dim]}>
          <Text style={s.section} onLongPress={transcribeTestFile}>Tande</Text>
          <Text style={s.hint}>Kenbe bouton an, pale an kreyòl, epi lage l.</Text>
          <Pressable
            disabled={phase !== 'ready' || hearing}
            onPressIn={startListening}
            onPressOut={stopListening}
            style={[s.mic, listening && s.micOn]}>
            {hearing ? <ActivityIndicator color="#fff" /> : <Text style={s.micText}>{listening ? 'Ap koute…' : 'Kenbe pou pale'}</Text>}
          </Pressable>
          {!!heard && <Text style={s.transcript}>{heard}</Text>}
          {!!heardInfo && <Text style={s.meta}>{heardInfo}</Text>}
        </View>

        <View style={[s.card, phase !== 'ready' && s.dim]}>
          <Text style={s.section} onLongPress={benchVoice}>Pale</Text>
          <TextInput value={text} onChangeText={setText} multiline style={s.input} placeholder="Ekri yon tèks an kreyòl…" placeholderTextColor="#5b6573" />
          <View style={s.voices}>
            {VOICES.map((v) => (
              <Pressable key={v.id} onPress={() => setVoice(v.id)} style={[s.chip, voice === v.id && s.chipOn]}>
                <Text style={[s.chipText, voice === v.id && s.chipTextOn]}>{v.label}</Text>
              </Pressable>
            ))}
          </View>
          <Pressable disabled={phase !== 'ready' || !text.trim()} onPress={speak} style={[s.primary, speaking && s.stop]}>
            <Text style={s.primaryText}>{speaking ? 'Kanpe' : 'Li l'}</Text>
          </Pressable>
          {!!spokeInfo && <Text style={s.meta}>{spokeInfo}</Text>}
        </View>
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0b0d10' },
  page: { padding: 16, paddingTop: 64, gap: 16 },
  title: { color: '#e8ecf1', fontSize: 30, fontWeight: '700' },
  sub: { color: '#8b95a3', fontSize: 14, marginTop: -10 },
  card: { backgroundColor: '#13161b', borderColor: '#22272f', borderWidth: 1, borderRadius: 16, padding: 16, gap: 12 },
  dim: { opacity: 0.45 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  section: { color: '#e8ecf1', fontSize: 18, fontWeight: '600' },
  hint: { color: '#8b95a3', fontSize: 13 },
  body: { color: '#c9d1db', fontSize: 14, flexShrink: 1 },
  error: { color: '#ff5a6e', fontSize: 13 },
  mic: { height: 96, borderRadius: 48, backgroundColor: '#1d4f7c', alignItems: 'center', justifyContent: 'center' },
  micOn: { backgroundColor: '#ff5a6e' },
  micText: { color: '#fff', fontSize: 17, fontWeight: '600' },
  transcript: { color: '#e8ecf1', fontSize: 17, lineHeight: 24 },
  meta: { color: '#8b95a3', fontSize: 12 },
  input: { minHeight: 110, color: '#e8ecf1', fontSize: 16, lineHeight: 22, backgroundColor: '#0f1216',
    borderColor: '#22272f', borderWidth: 1, borderRadius: 10, padding: 12, textAlignVertical: 'top' },
  voices: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { paddingHorizontal: 12, paddingVertical: 7, borderRadius: 14, borderWidth: 1, borderColor: '#2a313a' },
  chipOn: { backgroundColor: '#1d4f7c', borderColor: '#2c5a85' },
  chipText: { color: '#8b95a3', fontSize: 13 },
  chipTextOn: { color: '#fff' },
  primary: { backgroundColor: '#4aa3ff', borderRadius: 12, paddingVertical: 13, alignItems: 'center' },
  stop: { backgroundColor: '#ff5a6e' },
  primaryText: { color: '#0b0d10', fontSize: 16, fontWeight: '700' },
});
