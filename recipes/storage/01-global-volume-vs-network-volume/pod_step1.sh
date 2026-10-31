#!/usr/bin/env bash
# Plan step 1, run as the Pod's start command (EU-RO-1, NV at /workspace, GV at /workspace-global).
# Download both models to the network volume, verify sha256, copy to the global volume, make the
# small-file sets. Results print as "RB_RESULT {json}" lines so they can be read from the Pod logs.
# The Pod stays up afterwards (sleep) so the volumes can be inspected; delete it when done.
#
# Env: BENCH_TGZ_B64 (pod_step1.sh + read_bench.sh + manifests/, tar.gz, base64), HF_TOKEN (Runpod secret, for the
# private H3 repo). The token is never printed.
set -Euo pipefail

log() { echo "[step1 $(date -u +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; echo "RB_STEP1 failed"; sleep infinity; }
trap 'fail "line $LINENO"' ERR

QWEN_REPO=PassingByPixels/Qwen3.8-27B-NVFP4
QWEN_REV=1e5f29f3212294efdef42a873672c9bc399cae3c
H3_REPO=${H3_REPO:?set H3_REPO to your Hugging Face repo, e.g. org/name}
H3_REV=${H3_REV:?set H3_REV to the repo revision}
H3_REPO_DIR=${H3_REPO//\//--}
QWEN_DIR=/workspace/models/qwen38-27b-nvfp4
H3_SNAP=/workspace/h3-hub/models--${H3_REPO_DIR}/snapshots/$H3_REV

export RESULTS=/root/read_bench.jsonl SEEN=/root/.read_bench_seen
export HF_HUB_DISABLE_TELEMETRY=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0

log "mounts"; df -hT /workspace /workspace-global || true
[[ -d /workspace && -d /workspace-global ]] || fail "volumes not mounted"

log "install tools"
apt-get update -qq >/dev/null && apt-get install -y -qq rsync hostname >/dev/null
pip install -q -U "huggingface_hub[hf_xet]" >/dev/null 2>&1

# The start command already unpacked BENCH_TGZ_B64 into /root/bench (see README).
cd /root/bench || fail "no /root/bench"
chmod +x read_bench.sh

timed_download() {  # label repo rev dir [token]
  local label=$1 repo=$2 rev=$3 dir=$4 t0 t1 bytes
  mkdir -p "$dir"
  t0=$(date +%s.%N)
  hf download "$repo" --revision "$rev" --local-dir "$dir" --quiet >/dev/null
  t1=$(date +%s.%N)
  bytes=$(du -sb --exclude=.cache "$dir" | cut -f1)
  echo "RB_RESULT {\"kind\":\"download\",\"label\":\"$label\",\"repo\":\"$repo\",\"bytes\":$bytes,\"seconds\":$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}'),\"dc\":\"${RUNPOD_DC_ID:-}\"}"
}

# On a restart (env change resets the Pod) the NV already holds verified models: skip the download.
if [[ -f /workspace/.step1-downloaded ]]; then
  log "models already downloaded and verified on NV (marker found), skipping download"
else
  log "download qwen to NV"
  timed_download dl-qwen "$QWEN_REPO" "$QWEN_REV" "$QWEN_DIR"
  log "download h3 to NV"
  timed_download dl-h3 "$H3_REPO" "$H3_REV" "$H3_SNAP"
  log "verify on NV"
  ./read_bench.sh verify verify-qwen-nv "$QWEN_DIR" manifests/qwen38-27b-nvfp4.sha256
  ./read_bench.sh verify verify-h3-nv "$H3_SNAP" manifests/h3-models.sha256
  date -u +%FT%TZ >/workspace/.step1-downloaded
fi
unset HF_TOKEN

log "clear earlier GV copies, then copy NV -> GV (A1)"
rm -rf /workspace-global/models /workspace-global/h3-hub /workspace-global/bench-small
./read_bench.sh copy a1-qwen /workspace/models /workspace-global/models
./read_bench.sh copy a1-h3 /workspace/h3-hub /workspace-global/h3-hub

log "small-file sets"
[[ -d /workspace/bench-small ]] || ./read_bench.sh small-prepare /workspace
./read_bench.sh small-prepare /workspace-global

# Verify the GV copy last: it reads every GV file once, which is fine here because step 2 uses
# fresh Pods for the timed first reads.
log "verify on GV"
./read_bench.sh verify verify-qwen-gv "/workspace-global/models/qwen38-27b-nvfp4" manifests/qwen38-27b-nvfp4.sha256
./read_bench.sh verify verify-h3-gv "/workspace-global/h3-hub/models--${H3_REPO_DIR}/snapshots/$H3_REV" manifests/h3-models.sha256

for f in /root/read_bench_*_rsync_warnings.txt; do
  [[ -f "$f" ]] && { log "rsync warnings in $f (first 20 lines)"; head -20 "$f"; }
done
while read -r line; do echo "RB_RESULT $line"; done <"$RESULTS"
log "done"; echo "RB_STEP1 done"
sleep infinity
