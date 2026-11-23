#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 2 ]]; then
  echo "Usage: ./scripts/bench.sh /path/to/llama-bench /path/to/Kolibri-1.1-Q4_K_M.gguf" >&2
  exit 2
fi
BIN="$1"
MODEL="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/results"
for n in 50 30 0; do
  CUDA_VISIBLE_DEVICES=0 "$BIN" -m "$MODEL" -p 256 -n 96 -r 5 -ngl 999 -ncmoe "$n" -fitt 2048 -t 8 -o json > "$ROOT/results/q4-4090-ncmoe$n-clean-r5.json" 2> "$ROOT/results/q4-4090-ncmoe$n-clean-r5.log"
done
