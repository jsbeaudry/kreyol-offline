"""Push both converted models as private repos, one commit each (venv-whisper).

Files are staged as hard links next to the outputs, so staging costs no disk.
"""
import os
import shutil

from huggingface_hub import HfApi

from common import OUT_ASR, OUT_TTS, TTS_NAME, VOICES, W

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = {
    "jsbeaudry/oswald-large-v3-turbo-m3-ggml": (f"{W}/stage_asr", {
        "README.md": f"{HERE}/card_asr.md",
        **{f"ggml-oswald-m3-{q}.bin": f"{OUT_ASR}/ggml-oswald-m3-{q}.bin" for q in ("f16", "q8_0", "q5_0")},
        "ggml-silero-v6.2.0.bin": f"{OUT_ASR}/ggml-silero-v6.2.0.bin",
        "eval.json": f"{OUT_ASR}/eval.json",
    }),
    f"jsbeaudry/{TTS_NAME}-GGUF": (f"{W}/stage_tts", {
        "README.md": f"{HERE}/card_tts.md",
        **{f"{TTS_NAME}-{q}.gguf": f"{OUT_TTS}/{TTS_NAME}-{q}.gguf" for q in ("f16", "Q8_0", "Q4_K_M")},
        **{f"mmproj-{TTS_NAME}-{q}.gguf": f"{OUT_TTS}/mmproj-{TTS_NAME}-{q}.gguf" for q in ("f16", "Q8_0")},
        **{f"voices/{v}.wav": f"{OUT_TTS}/voices/{v}.wav" for v in VOICES},
        "voices/voices.json": f"{OUT_TTS}/voices/voices.json",
        "say.py": f"{HERE}/say.py",
        "kreyol_text.py": f"{HERE}/kreyol_text.py",
        "eval/eval_12_sentences.json": f"{OUT_TTS}/eval.json",
        "eval/eval_36_sentences.json": f"{OUT_TTS}/eval-big.json",
    }),
}

api = HfApi()
for repo, (stage, files) in REPOS.items():
    shutil.rmtree(stage, ignore_errors=True)
    for dst, src in files.items():
        path = os.path.join(stage, dst)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            os.link(src, path)
        except OSError:        # different filesystem (the cards live on /workspace)
            shutil.copy(src, path)
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=stage, commit_message="Add offline GGUF/ggml conversion, measured")
    info = api.model_info(repo, files_metadata=True)
    total = sum(s.size or 0 for s in info.siblings)
    print(f"https://huggingface.co/{repo}  private={info.private}  {len(info.siblings)} files, {total / 1e9:.2f} GB",
          flush=True)
