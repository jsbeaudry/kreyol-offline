"""validate.py for the append-only KV LM export: same 20 held-out sentences, scored with score.py.

    MAIN=path/to/flow_lm_main_kv_*.onnx VOICE=voices/reel.npy .venv/bin/python validate_kv.py BUNDLE_DIR OUT_DIR
"""
import json, os, sys, time
import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import pocket_tts_onnx  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TOKENIZER = os.path.expanduser("~/kreyol-offline/models/pocket-kreyol/tokenizer.json")
VOICE = os.environ.get("VOICE") or "voices/reel.npy"


class OurTokenizer:
    """Our fine-tune's Unigram tokenizer behind the two sentencepiece calls the runtime makes."""
    def __init__(self, path):
        self.tk = Tokenizer.from_file(path)

    def Encode(self, text):
        return self.tk.encode(text).ids

    def Decode(self, ids):
        return self.tk.decode(list(ids))


class KV(pocket_tts_onnx.PocketTTSOnnx):
    def _load_models(self):   # the phone's INT8 files; no voice encoder (voices are embeddings)
        opts = self._make_session_options()
        load = lambda name: ort.InferenceSession(str(self.bundle_dir / name), sess_options=opts, providers=self.providers)
        self.text_conditioner = load("text_conditioner_int8.onnx")
        self.flow_lm_main = ort.InferenceSession(os.environ["MAIN"], sess_options=opts, providers=self.providers)
        self.flow_lm_flow = load("flow_lm_flow_int8.onnx")
        self.mimi_decoder = load("mimi_decoder_int8.onnx")

    def _past(self):
        return {f"past_{kv}_{i}": np.zeros((1, 0, 16, 64), np.float32) for i in range(24) for kv in "kv"}

    def _append(self, past, result):
        names = [o.name for o in self.flow_lm_main.get_outputs()]
        out = dict(zip(names, result))
        return {n: np.concatenate([v, out[n.replace("past", "new")]], 1) for n, v in past.items()}

    def _condition_with_voice_embeddings(self, embeddings):
        past = self._past()
        empty = np.zeros((1, 0, self.latent_dim), np.float32)
        result = self.flow_lm_main.run(None, {"sequence": empty, "text_embeddings": self._prepare_voice_embeddings(embeddings), **past})
        return self._append(past, result)

    def _run_flow_lm_chunk(self, initial_state, text_ids, max_frames, frames_after_eos):
        past = dict(initial_state)
        text = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text.ndim == 2:
            text = text[None]
        empty_seq = np.zeros((1, 0, self.latent_dim), np.float32)
        empty_text = np.zeros((1, 0, self.conditioning_dim), np.float32)
        past = self._append(past, self.flow_lm_main.run(None, {"sequence": empty_seq, "text_embeddings": text, **past}))
        curr = np.full((1, 1, self.latent_dim), np.nan, np.float32)
        eos_step = None
        for step in range(max_frames or self._estimate_max_gen_len(text_ids.shape[1])):
            result = self.flow_lm_main.run(None, {"sequence": curr, "text_embeddings": empty_text, **past})
            conditioning, eos = result[0], result[1]
            past = self._append(past, result)
            if eos[0][0] > -4.0 and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            x = np.random.normal(0.0, np.sqrt(self.temperature), (1, self.latent_dim)).astype(np.float32)
            for s_arr, t_arr in self._st_buffers:
                x = x + self.flow_lm_flow.run(None, {"c": conditioning, "s": s_arr, "t": t_arr, "x": x})[0] / self.lsd_steps
            curr = x.reshape(1, 1, self.latent_dim)
            yield curr


np.random.seed(int(os.environ.get("SEED", "0")))
bundle, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
lines = json.load(open("val/onnx_int8/lines.json"))
tts = KV(models_dir=os.path.dirname(bundle), language=os.path.basename(bundle), precision="int8", tokenizer_path="cfg/onnx/french_24l/tokenizer.model")
tts.tokenizer = OurTokenizer(TOKENIZER)
voice = np.load(VOICE) if VOICE.endswith(".npy") else VOICE
spent = heard = 0.0
for i, text in enumerate(lines):
    t = time.time()
    audio = np.asarray(tts.generate(text, voice), np.float32)
    spent += time.time() - t; heard += len(audio) / 24000
    sf.write(f"{out}/{i:02d}.wav", audio, 24000)
print(f"{out}: RTF {spent / heard:.2f}")
