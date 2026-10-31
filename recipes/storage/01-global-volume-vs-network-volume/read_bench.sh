#!/usr/bin/env bash
# Disk measurements inside a Runpod Pod: network volume vs global volume.
# See docs/global-volume-v1/benchmark-plan.md, steps 1-2 (A1, A2, A3).
#
#   read_bench.sh copy   <label> <src_dir> <dst_dir>         # A1: rsync time (writes dst)
#   read_bench.sh verify <label> <root_dir> <manifest.sha256> # sha256 check after download/copy
#   read_bench.sh small-prepare <root_dir>                    # make bench-small/: 1,000 x 100KB files
#   read_bench.sh run    <label> <name>=<root_dir> [...]      # A2 + A3: first reads in this Pod
#
# Every measurement prints one JSON line and appends it to $RESULTS (default /root/read_bench.jsonl,
# on the container disk, so writing results never touches the volumes being measured).
# Copy that file out after each Pod (console file browser or `runpodctl send`).
#
# A2 only means something on the first read of a file in a fresh Pod: the page cache is not
# droppable inside a container. `run` refuses to read a file twice in the same Pod unless FORCE=1.
set -Eeuo pipefail

RESULTS="${RESULTS:-/root/read_bench.jsonl}"
SEEN="${SEEN:-/root/.read_bench_seen}"
BS="${BS:-16M}"
SMALL_COUNT="${SMALL_COUNT:-1000}"
SMALL_KB="${SMALL_KB:-100}"

# Large files read in A2, relative to each volume root (layout from benchmark-plan.md step 1).
H3_REV=${H3_REV:-}          # your H3 model repo revision (only for the H3 file)
H3_REPO=${H3_REPO:-}        # e.g. org/name
H3_REPO_DIR=${H3_REPO//\//--}
LARGE_FILES=(
  "models/qwen38-27b-nvfp4/model-00001-of-00003.safetensors"
  "h3-hub/models--${H3_REPO_DIR}/snapshots/${H3_REV}/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
)
[[ -n "${LARGE_FILES_OVERRIDE:-}" ]] && IFS=',' read -r -a LARGE_FILES <<<"$LARGE_FILES_OVERRIDE"

die() { echo "read_bench: $*" >&2; exit 1; }

now() {  # seconds with sub-second precision; bash 5 has EPOCHREALTIME, otherwise python3
  if [[ -n "${EPOCHREALTIME:-}" ]]; then echo "${EPOCHREALTIME/,/.}"; else python3 -c 'import time; print(f"{time.time():.6f}")'; fi
}
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }
mbps() { awk -v bytes="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.1f", bytes / 1e6 / s; else print "null" }'; }
fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }
jstr() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1" 2>/dev/null || printf '"%s"' "${1//\"/\\\"}"; }

context() {  # who/where this Pod is, as JSON fields (no secrets: only ids and hardware)
  local gpu
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  printf '"pod_id":%s,"dc":%s,"host":%s,"gpu":%s,"recorded_at":%s' \
    "$(jstr "${RUNPOD_POD_ID:-}")" "$(jstr "${RUNPOD_DC_ID:-}")" "$(jstr "${HOSTNAME:-$(cat /etc/hostname 2>/dev/null)}")" \
    "$(jstr "$gpu")" "$(jstr "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
}

fstype() { df -PT "$1" 2>/dev/null | awk 'NR==2 { print $2 }' || true; }

emit() { echo "$1" | tee -a "$RESULTS"; }

cmd_copy() {
  local label="$1" src="${2%/}" dst="${3%/}"
  [[ -d "$src" ]] || die "no source dir $src"
  mkdir -p "$dst"
  local bytes t0 t1 s warn_file warnings rc=0
  bytes=$(du -sb --exclude=.cache "$src" 2>/dev/null | awk '{ print $1 }' || du -sk "$src" | awk '{ print $1 * 1024 }')
  warn_file=$(mktemp)
  t0=$(now)
  # -rl --inplace instead of -a: a global volume cannot set permission bits, owners or mtimes
  # (rsync exits 23 on 2026-10-22), and has no atomic rename, so skip rsync's temp-file + rename.
  # .cache is Hugging Face download metadata, not model data.
  rsync -rl --inplace --exclude=.cache "$src/" "$dst/" 2>"$warn_file" || rc=$?
  t1=$(now)
  if [[ $rc -ne 0 && $rc -ne 23 ]]; then cat "$warn_file" >&2; die "rsync failed ($rc)"; fi
  s=$(elapsed "$t0" "$t1")
  warnings=$(wc -l <"$warn_file" | tr -d ' ')
  [[ "$warnings" -gt 0 ]] && cp "$warn_file" "/root/read_bench_${label}_rsync_warnings.txt"
  rm -f "$warn_file"
  emit "{\"kind\":\"copy\",\"label\":$(jstr "$label"),\"src\":$(jstr "$src"),\"dst\":$(jstr "$dst"),\"dst_fstype\":$(jstr "$(fstype "$dst")"),\"bytes\":$bytes,\"seconds\":$s,\"mb_per_s\":$(mbps "$bytes" "$s"),\"rsync_exit\":$rc,\"rsync_warning_lines\":$warnings,$(context)}"
}

cmd_verify() {
  local label="$1" root="${2%/}" manifest="$3" t0 t1 status=ok out
  [[ -f "$manifest" ]] || die "no manifest $manifest"
  manifest=$(cd "$(dirname "$manifest")" && pwd)/$(basename "$manifest")
  t0=$(now)
  out=$(cd "$root" && sha256sum -c "$manifest" 2>&1) || status=failed
  t1=$(now)
  echo "$out" >&2
  emit "{\"kind\":\"verify\",\"label\":$(jstr "$label"),\"root\":$(jstr "$root"),\"manifest\":$(jstr "$(basename "$manifest")"),\"status\":\"$status\",\"seconds\":$(elapsed "$t0" "$t1"),$(context)}"
  [[ "$status" == ok ]]
}

cmd_small_prepare() {
  local root="${1%/}" dir i
  dir="$root/bench-small"
  [[ -e "$dir" ]] && die "$dir exists; delete it first if you really want to recreate it"
  mkdir -p "$dir"
  for ((i = 0; i < SMALL_COUNT; i++)); do
    head -c $((SMALL_KB * 1024)) /dev/urandom >"$(printf '%s/f%04d.bin' "$dir" "$i")"
  done
  echo "made $SMALL_COUNT files of ${SMALL_KB}KB in $dir" >&2
}

read_large() {  # A2: sequential first read of one file
  local label="$1" name="$2" root="$3" rel="$4" path bytes t0 t1 s key
  path="$root/$rel"
  [[ -f "$path" ]] || { echo "skip missing $path" >&2; return 0; }
  key="$(realpath "$path" 2>/dev/null || echo "$path")"
  if grep -qxF "$key" "$SEEN" 2>/dev/null && [[ "${FORCE:-0}" != 1 ]]; then
    die "$path was already read in this Pod (page cache). Use a fresh Pod, or FORCE=1 to record a warm read."
  fi
  bytes=$(fsize "$path")
  t0=$(now)
  dd if="$path" of=/dev/null bs="$BS" status=none
  t1=$(now)
  echo "$key" >>"$SEEN"
  s=$(elapsed "$t0" "$t1")
  emit "{\"kind\":\"read_large\",\"label\":$(jstr "$label"),\"volume\":$(jstr "$name"),\"fstype\":$(jstr "$(fstype "$root")"),\"file\":$(jstr "$(basename "$rel")"),\"bytes\":$bytes,\"seconds\":$s,\"mb_per_s\":$(mbps "$bytes" "$s"),\"warm_read\":$([[ "${FORCE:-0}" == 1 ]] && echo true || echo false),$(context)}"
}

read_small() {  # A3: list + read 1,000 small files
  local label="$1" name="$2" root="$3" dir t0 t1 t2 n bytes
  dir="$root/bench-small"
  [[ -d "$dir" ]] || { echo "skip: no $dir (run small-prepare first)" >&2; return 0; }
  t0=$(now)
  n=$(find "$dir" -type f | wc -l | tr -d ' ')
  t1=$(now)
  bytes=$(find "$dir" -type f -exec cat {} + | wc -c | tr -d ' ')
  t2=$(now)
  emit "{\"kind\":\"read_small\",\"label\":$(jstr "$label"),\"volume\":$(jstr "$name"),\"fstype\":$(jstr "$(fstype "$root")"),\"files\":$n,\"bytes\":$bytes,\"list_seconds\":$(elapsed "$t0" "$t1"),\"read_seconds\":$(elapsed "$t1" "$t2"),$(context)}"
}

cmd_run() {
  local label="$1"; shift
  [[ $# -ge 1 ]] || die "run needs at least one <name>=<root_dir>"
  local spec name root rel
  for spec in "$@"; do
    name="${spec%%=*}"; root="${spec#*=}"; root="${root%/}"
    [[ "$name" != "$spec" && -d "$root" ]] || die "bad target '$spec' (want name=/existing/dir)"
    for rel in "${LARGE_FILES[@]}"; do read_large "$label" "$name" "$root" "$rel"; done
    read_small "$label" "$name" "$root"
  done
}

case "${1:-}" in
  copy) shift; [[ $# -eq 3 ]] || die "usage: copy <label> <src_dir> <dst_dir>"; cmd_copy "$@" ;;
  verify) shift; [[ $# -eq 3 ]] || die "usage: verify <label> <root_dir> <manifest.sha256>"; cmd_verify "$@" ;;
  small-prepare) shift; [[ $# -eq 1 ]] || die "usage: small-prepare <root_dir>"; cmd_small_prepare "$@" ;;
  run) shift; [[ $# -ge 2 ]] || die "usage: run <label> <name>=<root_dir> [...]"; cmd_run "$@" ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
