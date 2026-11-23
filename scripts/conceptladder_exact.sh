#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
  echo "Usage: $0 /path/to/Kolibri-1.1-Q4_K_M.gguf /path/to/llama-server /path/to/conceptladder-bench /path/to/output-dir" >&2
  exit 2
fi
MODEL=$(realpath "$1")
SERVER=$(realpath "$2")
BENCH=$(realpath "$3")
OUTPUT=$4
PORT=${PORT:-18321}
mkdir -p "$OUTPUT"
OUTPUT=$(realpath "$OUTPUT")
if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Port $PORT is already in use" >&2
  exit 3
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$SERVER" -m "$MODEL" --alias kolibri11-q4 --host 127.0.0.1 --port "$PORT" -c 4096 -np 1 -ngl 999 -ncmoe 50 -fitt 2048 -t 8 --seed 17 --jinja --reasoning off --no-webui > "$OUTPUT/server.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT
ready=0
for (( i=0;i<240;i++ )); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "llama-server exited during startup" >&2
    exit 4
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
    ready=1
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  echo "llama-server startup timed out" >&2
  exit 5
fi
(
  cd "$BENCH"
  python3 -m conceptladder validate --suite exact
  for seed in 17 41; do
    python3 -m conceptladder run --suite exact --seed "$seed" --variants 4 --url "http://127.0.0.1:$PORT/v1" --model kolibri11-q4 --max-tokens 64 --timeout 120 --out "$OUTPUT/kolibri11-q4-seed${seed}.json" 2> "$OUTPUT/seed${seed}.progress"
    python3 -m conceptladder score "$OUTPUT/kolibri11-q4-seed${seed}.json" --out "$OUTPUT/kolibri11-q4-seed${seed}-score.json"
  done
)
sha256sum "$MODEL" > "$OUTPUT/model.sha256"
