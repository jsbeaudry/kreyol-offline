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
# T4, A100, A10G and L4 — the GPUs a Hugging Face endpoint is likely to give this.
# GGML_NATIVE=OFF below is not a tuning choice. Built with the default, ggml compiles for whatever the
# build machine happens to support, and a GitHub runner supports instructions a serverless host does
# not: whisper-server read the model header and then died with signal 4, SIGILL, an illegal
# instruction. The binary has to run on a CPU it was not built on, so it cannot be built for this one.
ARG CUDA_ARCHS="75;80;86;89"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The driver library is not in a devel image — only a stub, which is all a link needs. Without this the
# build spends forty minutes compiling and then dies with "undefined reference to cuGetErrorString".
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

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout ${LLAMA_COMMIT}

# The job loop: `-p -` keeps the model loaded and reads one job per stdin line. Without it every
# sentence would pay the model load, which is most of the cost of a short reading.
COPY llama-tts-serve.patch /src/llama-tts-serve.patch
RUN git apply --verbose /src/llama-tts-serve.patch

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-tts -j"$(nproc)"

# Fail the build rather than the first request if the patch silently did not take effect.
RUN test -x build/bin/llama-tts \
    && grep -q '@@tts' build/bin/llama-tts || (echo "the serve patch is not in this binary" && exit 1)


FROM nvidia/cuda:${CUDA}-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

COPY --from=build /src/build/bin/llama-tts /opt/llama/bin/llama-tts
COPY --from=build /src/build/bin/*.so /opt/llama/lib/
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
