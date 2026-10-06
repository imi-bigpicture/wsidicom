#!/usr/bin/env bash
# Default benchmark/correctness run for the Dockerfile.bench image.
# Override by passing your own command to `docker run`.
set -euo pipefail

echo "== CPU =="
lscpu | grep -iE 'model name|^flags' | head -2 || true

echo "== roofline (single-core bandwidth ceiling) =="
python bench/roofline.py

echo "== AVX-512 vs AVX2 =="
python bench/bench_avx512.py 16

echo "== correctness =="
python -m pytest tests/ -q
python tests/nets/oracle.py
python tests/nets/fuzz.py 400 71
