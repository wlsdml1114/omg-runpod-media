#!/usr/bin/env bash
# Install Ming-3.0-tts 0.5B on a Runpod Pod at the code commit and model revision used in this recipe.
# usage: bash scripts/setup_pod.sh            (WORK_ROOT defaults to /workspace)
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
MING_REPO="https://github.com/inclusionAI/Ming-3.0-tts.git"
MING_COMMIT="028c1eace193e7acdb256453da89b6e9207e052b"   # main on 2026-10-22 (experiment day)
MODEL_ID="inclusionAI/Ming-3.0-tts-0.5B"
MODEL_REV="8d66b70a00f05ad183fa8a863b4ce609fc83cbc0"     # Hugging Face main on 2026-10-22
MODEL_SHA256="45c4d78d8ad6b1d942e3da86ba035fece02de569f7b2ca7e9d121dfaf18d674c"  # model.safetensors
CAMPPLUS_SHA256="a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"

REPO_DIR="$WORK_ROOT/Ming-3.0-tts"
VENV="$WORK_ROOT/venv-ming"
MODEL_DIR="$WORK_ROOT/models/Ming-3.0-tts-0.5b"

echo "== 1/4 Ming code ($MING_COMMIT)"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$MING_REPO" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --quiet origin
git -C "$REPO_DIR" checkout --quiet "$MING_COMMIT"
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$MING_COMMIT"

echo "== 2/4 Python venv (torch 2.6.0 + cu124, Ming's pinned stack)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_DIR/requirements.txt"
if ! python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.6.0') and torch.cuda.is_available() else 1)"; then
  echo "torch is not 2.6.0 with CUDA; reinstalling the cu124 wheels"
  pip install --quiet torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== 3/4 Model weights ($MODEL_ID @ $MODEL_REV)"
pip install --quiet "huggingface_hub[cli]"
hf download "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR" --exclude "*.pdf" "figures/*"
echo "$MODEL_SHA256  $MODEL_DIR/model.safetensors" | sha256sum -c -
echo "$CAMPPLUS_SHA256  $MODEL_DIR/campplus.onnx" | sha256sum -c -

echo "== 4/4 Ming preflight"
(cd "$REPO_DIR" && python tools/preflight.py --model_path "$MODEL_DIR")

echo
echo "done. next:"
echo "  source $VENV/bin/activate"
echo "  python scripts/clone_ko.py --repo $REPO_DIR --model-path $MODEL_DIR --ref-wav ref/my_voice.wav --ref-text-file ref/my_voice.txt --text '테스트 문장이에요.'"
