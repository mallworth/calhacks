#!/usr/bin/env python3
"""
Fetch the MLX model repo locally into the iOS bundle folder so it can be copied at build/runtime.

Usage:
  python tools/fetch_mlx_model.py

Requirements:
  pip install huggingface_hub

It will download `mlx-community/Phi-3-mini-4k-instruct-4bit` into:
  app/ios/Runner/LLMModels/Phi-3-mini-4k-instruct-4bit

Note: Large models will make your .ipa bigger. For development or demos this is fine.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "mlx-community/Phi-3-mini-4k-instruct-4bit"

ROOT = Path(__file__).resolve().parents[1]  # /.../app
DEST = ROOT / "ios" / "Runner" / "LLMModels" / "Phi-3-mini-4k-instruct-4bit"

if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} to: {DEST}")
    tmp_dir = snapshot_download(repo_id=REPO_ID, local_dir=DEST, repo_type="model")
    print("Done.")
