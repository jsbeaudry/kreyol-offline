#!/bin/bash
# m3 (HF Whisper) -> whisper.cpp ggml at f16, q8_0 and q5_0, plus whisper.cpp's Silero VAD model.
set -eo pipefail
source "$(dirname "$0")/podenv.sh"
source /workspace/venv-whisper/bin/activate
W=/root/work; OUT=$W/out/asr; mkdir -p $OUT
hf download jsbeaudry/oswald-large-v3-turbo-m3 config.json generation_config.json model.safetensors \
  processor_config.json tokenizer.json tokenizer_config.json --local-dir $W/m3 > /dev/null
# whisper.cpp's converter reads vocab.json + added_tokens.json, which transformers 5 no longer saves.
# Fine-tuning never touched the tokenizer, so take them from the base model after proving they match.
hf download openai/whisper-large-v3-turbo vocab.json added_tokens.json --local-dir $W/turbo-tok > /dev/null
python - <<'PY'
import json
tj = json.load(open("/root/work/m3/tokenizer.json"))
vocab = json.load(open("/root/work/turbo-tok/vocab.json"))
added = json.load(open("/root/work/turbo-tok/added_tokens.json"))
m3_vocab = tj["model"]["vocab"]
m3_added = {t["content"]: t["id"] for t in tj["added_tokens"]}
assert m3_vocab == vocab, "BPE vocab differs from base turbo"
assert all(m3_added.get(k) == v for k, v in added.items()), "added tokens differ from base turbo"
print(f"tokenizer identical to base turbo: {len(vocab):,} BPE + {len(added):,} added tokens")
PY
cp $W/turbo-tok/vocab.json $W/turbo-tok/added_tokens.json $W/m3/
python $W/whisper.cpp/models/convert-h5-to-ggml.py $W/m3 $W/openai-whisper $OUT
mv $OUT/ggml-model.bin $OUT/ggml-oswald-m3-f16.bin
QUANT=$(ls $W/whisper.cpp/build/bin/ | grep -m1 -E '^(whisper-)?quantize$')
for q in q8_0 q5_0; do
  $W/whisper.cpp/build/bin/$QUANT $OUT/ggml-oswald-m3-f16.bin $OUT/ggml-oswald-m3-$q.bin $q > /dev/null
done
ls -la $OUT
