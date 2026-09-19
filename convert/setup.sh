#!/bin/bash
# One-time pod setup: build whisper.cpp and llama.cpp for this GPU, and three venvs that must not share packages.
#   venv-convert  llama.cpp's HF->GGUF converter; its requirements pin their own (CPU) torch
#   venv-whisper  transformers 5 for m3 (PyTorch baseline, whisper.cpp's converter, CER of TTS output)
#   venv-tts      qwen-tts for the reference voices and the PyTorch TTS baseline
set -eo pipefail
W=/root/work
mkdir -p $W && cd $W
apt-get update -qq && apt-get install -y -qq build-essential cmake git ffmpeg > /dev/null
[ -d whisper.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/whisper.cpp
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
[ -d openai-whisper ] || git clone -q --depth 1 https://github.com/openai/whisper openai-whisper  # mel_filters.npz
echo "whisper.cpp $(git -C whisper.cpp rev-parse --short HEAD) | llama.cpp $(git -C llama.cpp rev-parse --short HEAD)"

# Build only for this card's architecture: a CUDA build for every arch takes several times longer.
export PATH=/usr/local/cuda/bin:$PATH   # nvcc ships with the image but is not on PATH
# nproc reports the host; the cgroup quota is what this container may actually use.
JOBS=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo 800000) / 100000 )); [ "$JOBS" -ge 1 ] || JOBS=8
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
cmake -S whisper.cpp -B whisper.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
  -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build whisper.cpp/build -j $JOBS > $W/build-whisper.log 2>&1
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build llama.cpp/build -j $JOBS --target llama-tts llama-quantize > $W/build-llama.log 2>&1
ls whisper.cpp/build/bin llama.cpp/build/bin

[ -d /workspace/venv-convert ] || python -m venv /workspace/venv-convert
/workspace/venv-convert/bin/pip install -q -U pip
# From inside the repo: the requirements files refer to ./gguf-py and each other by relative path.
(cd llama.cpp && /workspace/venv-convert/bin/pip install -q -r requirements/requirements-convert_hf_to_gguf.txt)

[ -d /workspace/venv-whisper ] || python -m venv --system-site-packages /workspace/venv-whisper
/workspace/venv-whisper/bin/pip install -q -U pip
/workspace/venv-whisper/bin/pip install -q "transformers>=5.17,<6" jiwer soundfile librosa pyarrow "huggingface_hub[hf_xet]"

[ -d /workspace/venv-tts ] || python -m venv --system-site-packages /workspace/venv-tts
/workspace/venv-tts/bin/pip install -q -U pip
/workspace/venv-tts/bin/pip install -q "qwen-tts==0.1.1" soundfile librosa

/workspace/venv-whisper/bin/python -c "import torch, transformers; print('venv-whisper: torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
/workspace/venv-tts/bin/python -c "import torch, transformers, qwen_tts; print('venv-tts: torch', torch.__version__, '| transformers', transformers.__version__)"
/workspace/venv-convert/bin/python -c "import torch, transformers; print('venv-convert: torch', torch.__version__, '| transformers', transformers.__version__)"
