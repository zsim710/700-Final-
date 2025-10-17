#!/usr/bin/env python3
"""
Fetch the official SpeechBrain SA tokenizer by loading a pretrained ASR model once.

This script:
- Creates a fresh savedir for the pretrained model
- Loads the model via EncoderDecoderASR.from_hparams (which triggers a download if needed)
- Locates the tokenizer asset within savedir/save (resolving symlinks)
- Copies the tokenizer file to XDED/tokenizers/sa_official/

Default model: speechbrain/asr-transformer-transformerlm-librispeech

Outputs:
- Prints the resolved tokenizer source path and the copied destination path
"""

import os
import sys
import shutil
from pathlib import Path


def main():
    try:
        from speechbrain.inference.ASR import EncoderDecoderASR
    except Exception as e:
        print("ERROR: Failed to import SpeechBrain (speechbrain.inference.ASR).\n"
              "Make sure 'speechbrain' is installed or available on PYTHONPATH.")
        raise

    # Config
    repo_id = os.environ.get(
        "SB_ASR_REPO",
        "speechbrain/asr-transformer-transformerlm-librispeech",
    )
    workspace_root = Path(__file__).resolve().parents[2]
    savedir = workspace_root / "pretrained_models" / repo_id.split("/")[-1]
    save_subdir = savedir / "save"
    dest_dir = workspace_root / "tokenizers" / "sa_official"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using repo: {repo_id}")
    print(f"Savedir: {savedir}")
    # Ensure a clean savedir without clobbering existing downloads
    savedir.mkdir(parents=True, exist_ok=True)

    # Trigger download/load
    print("Loading pretrained model to trigger asset fetch (first run may take a while)...")
    asr = EncoderDecoderASR.from_hparams(source=repo_id, savedir=str(savedir))
    # Touch the tokenizer to ensure it's materialized
    _ = getattr(asr, "tokenizer", None)

    # Inspect savedir (root) and optionally savedir/save for tokenizer assets
    search_dirs = [savedir]
    if save_subdir.exists():
        search_dirs.append(save_subdir)

    candidates = []
    for d in search_dirs:
        for p in d.iterdir():
            name = p.name.lower()
            if any(k in name for k in ["token", "spm", "sentencepiece"]):
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "No tokenizer-like files found under savedir. Contents: "
            + ", ".join(
                sorted(
                    f"{d.name}:{x.name}" for d in search_dirs for x in d.iterdir()
                )
            )
        )

    # Prefer .model, else .ckpt, else any match
    def score(p: Path) -> int:
        s = p.name.lower()
        if s.endswith(".model"):
            return 3
        if s.endswith(".ckpt"):
            return 2
        if s.endswith(".zip") or s.endswith(".pt"):
            return 1
        return 0

    candidates.sort(key=score, reverse=True)
    chosen = candidates[0]

    # Resolve symlink to the real cache file
    resolved_src = Path(os.path.realpath(str(chosen)))
    print(f"Found tokenizer asset: {chosen} -> {resolved_src}")

    # Decide destination filename
    if resolved_src.name.lower().endswith(".model"):
        dest_name = "tokenizer.model"
    elif resolved_src.name.lower().endswith(".ckpt"):
        dest_name = "tokenizer.ckpt"
    else:
        # Keep original extension for safety
        dest_name = "tokenizer" + resolved_src.suffix

    dest_path = dest_dir / dest_name

    # Copy file
    shutil.copyfile(resolved_src, dest_path)
    print(f"Copied tokenizer to: {dest_path}")

    # Also copy hyperparams.yaml for reference if present
    hyp_path = savedir / "hyperparams.yaml"
    if hyp_path.exists():
        shutil.copyfile(hyp_path, dest_dir / "hyperparams.yaml")
        print(f"Copied hyperparams.yaml to: {dest_dir / 'hyperparams.yaml'}")

    print("Done. You can now point eval to this tokenizer path.")
    print("Example: --spm_model_path", dest_path if dest_path.suffix == ".model" else "(use --tokenizer_ckpt)" )


if __name__ == "__main__":
    sys.exit(main())
