# Kreyòl speech on a GPU: llama-tts and whisper.cpp in one worker.
#
# Why a GPU at all: measured on an M-series Mac, synthesis costs 1.05 s plus 0.74 s per second of
# audio, and that is most of every turn in a spoken conversation. Transcription adds a flat 0.9 s.
# Neither is the model being large; both are the machine.
#
# llama.cpp and whisper.cpp are pinned to the commits the local kit was built from. A different commit
# is a different voice — sampling defaults, the tokeniser and the codec path all move between releases.

ARG CUDA=12.4.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu22.04 AS build

ARG LLAMA_COMMIT=1af554f8fc78ba029665a47b839484d9763e2a75
ARG CUDA_ARCHS="75;89"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The driver library is not in a devel image — only a stub, which is all a link needs. Without this the
# build dies with "undefined reference to cuGetErrorString" after forty minutes of compiling, which is
# how the first attempt at this image was spent.
# A CUDA devel image ships a stub libcuda, not the driver, and nothing links against it by default.
# The first attempt at this put the stubs directory on the search path and stopped there — which tells
# the linker where to look without telling it what to look for, so the build still failed after fifty
# minutes with "undefined reference to cuMemMap". The library has to be named as well as found: hence
# the -lcuda below, and the stub placed where ldconfig can see it.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && echo /usr/local/cuda/lib64/stubs > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

# Prove the driver API links before spending fifty minutes finding out it does not. These are the exact
# symbols the real build failed on twice; if this cannot link them, neither can ggml-cuda, and the
# build should say so in seconds rather than after the whole compile.
RUN printf '%s\n' '#include <cuda.h>' 'int main(){CUdevice d;int v;' \
      'cuDeviceGetAttribute(&v,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,d);' \
      'cuMemGetAllocationGranularity(0,0,CU_MEM_ALLOC_GRANULARITY_MINIMUM);' \
      'const char*s;cuGetErrorString(CUDA_SUCCESS,&s);return 0;}' > /tmp/link-check.c \
    && gcc /tmp/link-check.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda \
         -o /tmp/link-check \
    && echo "the CUDA driver API links" \
    && rm -f /tmp/link-check /tmp/link-check.c

WORKDIR /llama
RUN git clone https://github.com/ggml-org/llama.cpp.git . && git checkout ${LLAMA_COMMIT}
COPY llama-tts-serve.patch /llama/serve.patch
RUN git apply --verbose /llama/serve.patch
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-tts -j"$(nproc)" \
    && grep -q '@@tts' build/bin/llama-tts || (echo "the serve patch is not in this binary" && exit 1)

WORKDIR /whisper
RUN git clone https://github.com/ggml-org/whisper.cpp.git . \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" -DWHISPER_BUILD_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target whisper-server -j"$(nproc)"


FROM nvidia/cuda:${CUDA}-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir runpod numpy

COPY --from=build /llama/build/bin/llama-tts /opt/llama/bin/
COPY --from=build /llama/build/bin/*.so /opt/llama/lib/
COPY --from=build /whisper/build/bin/whisper-server /opt/whisper/bin/
COPY --from=build /whisper/build/src/*.so* /whisper/build/ggml/src/*.so* /opt/whisper/lib/
ENV LD_LIBRARY_PATH=/opt/llama/lib:/opt/whisper/lib

COPY voices/ /opt/voices/
COPY kreyol_text.py handler.py /app/

# The models are private, so they are pulled at build time with a token the workflow supplies as a
# secret and this image never stores. Baked in rather than fetched at start: a cold worker that has to
# download two gigabytes before its first word is not an improvement on a slow laptop.
# A BuildKit secret, not an ARG: an ARG is recorded in the image history, so `docker history` on the
# pushed image would hand the token to anyone who can pull it. Mounted here, it exists only for the
# length of this one command and never reaches a layer.
RUN --mount=type=secret,id=HF_TOKEN \
    pip3 install --no-cache-dir "huggingface_hub[cli]" \
    && mkdir -p /models \
    && export HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" \
    && hf download jsbeaudry/qwen3-tts-1.7b-kreyol-GGUF \
         qwen3-tts-1.7b-kreyol-Q4_K_M.gguf mmproj-qwen3-tts-1.7b-kreyol-Q8_0.gguf \
         --local-dir /models \
    && hf download jsbeaudry/oswald-large-v3-turbo-m3-ggml \
         ggml-oswald-m3-q5_0.bin --local-dir /models \
    && unset HF_TOKEN \
    && pip3 uninstall -y huggingface_hub \
    && rm -rf /root/.cache/huggingface \
    && ls -la /models

ENV MODEL_DIR=/models VOICE_DIR=/opt/voices PYTHONUNBUFFERED=1
CMD ["python3", "-u", "/app/handler.py"]
