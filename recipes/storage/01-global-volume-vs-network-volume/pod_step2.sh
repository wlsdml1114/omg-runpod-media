#!/usr/bin/env bash
# Plan step 2, run as the start command of a fresh Pod: first reads only (A2 large files, A3 small files).
# Env: BENCH_TGZ_B64 (unpacked into /root/bench by the start command), ROUND (label, e.g. eu-r1),
#      TARGETS (space separated name=dir, read in this order, e.g. "gv=/workspace-global nv=/workspace").
# Results print as "RB_RESULT {json}". The Pod then sleeps; it is deleted from outside.
set -Euo pipefail
log() { echo "[step2 $(date -u +%H:%M:%S)] $*"; }
trap 'log "FAILED line $LINENO"; echo "RB_STEP2 failed"; sleep infinity' ERR

export RESULTS=/root/read_bench.jsonl SEEN=/root/.read_bench_seen
cd /root/bench || exit 1
chmod +x read_bench.sh
log "round $ROUND, targets: $TARGETS"
df -hT /workspace /workspace-global 2>/dev/null || true
# shellcheck disable=SC2086  # TARGETS is a word list on purpose
./read_bench.sh run "$ROUND" $TARGETS
while read -r line; do echo "RB_RESULT $line"; done <"$RESULTS"
log "done"; echo "RB_STEP2 done"
sleep infinity
