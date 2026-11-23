#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 2 ]]; then
  echo "Usage: ./scripts/bench_dual.sh /path/to/sm86-sm89/llama-bench /path/to/Kolibri-1.1-Q4_K_M.gguf" >&2
  exit 2
fi
BIN="$1"
MODEL="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/results"
CUDA_VISIBLE_DEVICES=0 "$BIN" -m "$MODEL" -p 256 -n 96 -r 5 -ngl 999 -ncmoe 30 -fitt 2048 -t 8 -o json > "$ROOT/results/q4-builddual-single-r5.json" 2> "$ROOT/results/q4-builddual-single-r5.log"
CUDA_VISIBLE_DEVICES=0,1 "$BIN" -m "$MODEL" -p 256 -n 96 -r 5 -ngl 999 -ncmoe 30 -ts 8/1 -fitt 2048 -t 8 -o json > "$ROOT/results/q4-builddual-two-gpus-r5.json" 2> "$ROOT/results/q4-builddual-two-gpus-r5.log"
