# Kolibri 1.1 Q4: independent 4090 test artifacts

This repository contains a small, reproducible **single-device** test of Aleph Alpha's official Kolibri 1.1 Q4_K_M GGUF, run November 23–24, 2026 (UTC / Singapore local date). No model weights are redistributed.

This is not a controlled BF16-versus-Q4 comparison or an evaluation of the vendor's 1M-token benchmarks. It does not establish EXL3 compatibility.

## Hardware and software

- NVIDIA RTX 4090 24 GB, AMD Ryzen 5 7600, 128 GB system RAM, Ubuntu
- Source GGUF: [Aleph-Alpha/Kolibri-1.1-GGUF](https://huggingface.co/Aleph-Alpha/Kolibri-1.1-GGUF)
- File: `Kolibri-1.1-Q4_K_M.gguf` (45,508,762,048 bytes)
- Model file SHA-256: `ee511452dceb9171e9b2b7c2e585327dbf097b5852fdd2ffbdc4ee1223d1ab10`
- Runtime: [llama.cpp PR #32891](https://github.com/ggml-org/llama.cpp/pull/32891) at `5f4d222898ddb12cf581edf1b7e841d80e0b2e17`, CUDA architecture SM89
- Each setting used 5 repetitions; prompt processing 256 tokens, generation 96 tokens, 8 CPU threads, `-ngl 999 -fitt 2048`; measured with `llama-bench`.
- Other GPU processes were stopped for the **clean** data reported below.

| RTX 4090 configuration | pp256 tokens/s | tg96 tokens/s |
|---|---:|---:|
| `-ncmoe 50` | 440.56 | 70.95 |
| `-ncmoe 30` | 440.19 | 70.71 |
| `-ncmoe 0` | 432.80 | 70.71 |

See the individual per-repetition timings in `results/*clean-r5.json`. Numeric timings are preserved; host-specific absolute model paths and instrument timestamp fields were normalized before public release.

The first two-repetition pilot measured 62–65 tokens/s. A subsequent attempt with a separately running 8k-context model server holding ~20GB VRAM measured ~18 tokens/s; **those contaminated measurements were excluded**. Killing the leftover server restored 70–71 tokens/s. This illustrates how model residency and memory placement can completely invalidate a seemingly fair performance comparison.

## Reproduce the measurements

Build the pinned llama.cpp PR with CUDA SM89, then run:

    ./scripts/bench.sh /path/to/llama-bench /path/to/Kolibri-1.1-Q4_K_M.gguf

The script measures only GPU index 0 as exposed by `CUDA_VISIBLE_DEVICES`. Shut down other model servers first. Do not compare runs with different resident models, CUDA build targets, prompt lengths, contexts, KV cache formats or GPU availability.

## Small application-level smoke suite

The 10 handcrafted sanity cases are available in `scripts/quality_probe.py` and `results/quality.json`. With a loopback-only server running:

    CUDA_VISIBLE_DEVICES=0 /path/to/llama-server -m /path/to/Kolibri-1.1-Q4_K_M.gguf --host 127.0.0.1 --port 19191 -c 8192 --n-gpu-layers 999 --n-cpu-moe 30 --fit on -t 8
    python3 scripts/quality_probe.py

Observed: **9/10** simple cases passed. Arithmetic, German arithmetic, valid JSON, a narrow synthetic prompt-injection example and 6.4k-token retrieval passed. The July sales citation question returned `Document B` but omitted the figure `21`, so failed the exact expected-answer check. This is a small handcrafted functionality probe **not** AgentDojo, not a factual accuracy benchmark, and not a general prompt-injection safety result. It has no matched FP8 or other-model baseline.

## Matched 4090 versus 4090+3060 experiment

We rebuilt the **same pinned llama.cpp commit** with both SM86 (RTX 3060) and SM89 (RTX 4090) CUDA targets, then ran five repeats for each configuration using the same Q4 checkpoint and `-ncmoe 30 -ngl 999 -fitt 2048 -t 8` settings. Dual-device runs used `-ts 8/1`.

| Setup | pp256 tokens/s | tg96 tokens/s |
|---|---:|---:|
| RTX 4090 alone (dual-target binary) | 408.29 | **70.52** |
| RTX 4090 + RTX 3060, split 8:1 | **577.22** | 60.23 |

The second card improved prefill by **41.4%** but reduced per-token decode by **14.6%** on this specific test. Results are in `results/q4-builddual-*.json`; reproduction is in `scripts/bench_dual.sh`. This is a single split, not a tuning sweep; do not assume dual GPUs always hurt or help.

The initial two-device attempt had aborted with `CUDA error: no kernel image is available for execution on the device` because we originally built for SM89 only. That failure was an **invalid CUDA build target**, corrected by compiling for `86;89`, not an actual Kolibri architecture limitation.

## Reuse

The scripts and metadata in this repository are MIT licensed. The model weights remain with the original publisher and subject to the original model license. Results are machine- and version-specific. No server should be exposed without authentication; the server example binds to loopback only.
