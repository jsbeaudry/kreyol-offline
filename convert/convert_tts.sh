#!/bin/bash
# qwen3-tts-1.7b-kreyol (the clone-capable base variant) -> llama.cpp GGUF: talker f16/Q8_0/Q4_K_M + mmproj f16/Q8_0.
# The mmproj holds the speaker-reference encoder and the codec decoder that turns codes into audio.
set -eo pipefail
source "$(dirname "$0")/podenv.sh"
W=/root/work; OUT=$W/out/tts; mkdir -p $OUT
N=qwen3-tts-1.7b-kreyol
/workspace/venv-whisper/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('jsbeaudry/$N', local_dir='$W/tts', ignore_patterns=['lora/*'])"
source /workspace/venv-convert/bin/activate
python $W/llama.cpp/convert_hf_to_gguf.py $W/tts --outtype f16 --outfile $OUT/$N-f16.gguf
python $W/llama.cpp/convert_hf_to_gguf.py $W/tts --mmproj --outtype f16 --outfile $OUT/mmproj-$N-f16.gguf
python $W/llama.cpp/convert_hf_to_gguf.py $W/tts --mmproj --outtype q8_0 --outfile $OUT/mmproj-$N-Q8_0.gguf
for q in Q8_0 Q4_K_M; do
  $W/llama.cpp/build/bin/llama-quantize $OUT/$N-f16.gguf $OUT/$N-$q.gguf $q > /dev/null
done
ls -la $OUT
