# The llama-tts binary the local kit runs, in an image, behind an HTTP port.
#
# Hugging Face's own llama.cpp container serves llama-server, which has no speech-synthesis route: a
# request to the deployed endpoint returns audio codec token ids and an empty `content`, because the
# vocoding step lives in llama-tts and llama-server never runs it. This image runs llama-tts instead.
#
# llama.cpp is pinned to the commit the local kit was built from. A different commit is a different
# voice: the sampling defaults, the tokeniser and the codec path all move between releases, and the
# point of this image is to sound like the local kit rather than merely to work.

ARG CUDA=12.4.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu22.04 AS build

# The commit ~/kreyol-offline/llama.cpp is checked out at.
ARG LLAMA_COMMIT=1af554f8fc78ba029665a47b839484d9763e2a75
# 86 is Ampere (A5000, RTX 3090, A4000, A40) and 89 is Ada (L4, RTX 2000 Ada, RTX 4090) — between
# them every GPU in RunPod's cheap serverless tier. An earlier default of "75;89" was wrong: 75 is the
# T4, which RunPod does not offer for serverless at all, and it left out Ampere entirely.
ARG CUDA_ARCHS="86;89"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout ${LLAMA_COMMIT}

# The job loop: `-p -` keeps the model loaded and reads one job per stdin line. Without it every
# sentence would pay the model load, which is most of the cost of a short reading.
COPY llama-tts-serve.patch /src/llama-tts-serve.patch
RUN git apply --verbose /src/llama-tts-serve.patch

# libggml-cuda calls the CUDA *driver* API (cuMemMap, cuDeviceGetAttribute), which lives in
# libcuda.so — a file that exists only on a machine with an NVIDIA driver. A build runner has no GPU,
# so the toolkit's stub has to be on the link path or the link fails with undefined `cu*` symbols.
# The stub is never copied into the runtime image: there the real driver arrives from the host.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

# GGML_NATIVE=OFF because the runner's CPU is not the deploy machine's. Left on, ggml compiles with
# -march=native for whatever the runner happens to be, and the image can then hit an illegal
# instruction on a different host.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build build --target llama-tts -j"$(nproc)"

# Fail the build rather than the first request if the patch silently did not take effect.
RUN test -x build/bin/llama-tts \
    && grep -q '@@tts' build/bin/llama-tts || (echo "the serve patch is not in this binary" && exit 1)


FROM nvidia/cuda:${CUDA}-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# The whole directory, symlinks and all: llama.cpp ships libllama.so, libllama.so.0 and
# libllama.so.0.24.0 as links to one file, and the binary looks the library up by its SONAME.
COPY --from=build /src/build/bin/ /opt/llama/bin/
RUN mkdir -p /opt/llama/lib && mv /opt/llama/bin/*.so* /opt/llama/lib/ \
    && find /opt/llama/bin -type f ! -name llama-tts -delete
ENV LD_LIBRARY_PATH=/opt/llama/lib

COPY voices/ /opt/voices/
COPY kreyol_text.py server.py /app/

# The endpoint mounts the model repository here; MODEL and MMPROJ override the search if needed.
ENV MODEL_DIR=/repository \
    VOICE_DIR=/opt/voices \
    LLAMA_TTS=/opt/llama/bin/llama-tts \
    PORT=80 \
    NGL=99 \
    PYTHONUNBUFFERED=1

EXPOSE 80
# The endpoint's readiness probe. /health answers 503 until llama-tts has said ready and warmed up,
# so traffic is not sent to a container that would take a minute to answer its first request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:80/health',timeout=4).status==200 else 1)"

CMD ["python3", "/app/server.py"]
