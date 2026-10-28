#!/usr/bin/env bash
# One erased model/intervention, paired I2P and MS-COCO 1K evaluation.
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

# 只有显式指定 --vlm-model /本地目录 时，才按本地路径加载；独立的 download 命令仅用于可选的提前下载。

export BENCHMARK_CALLER_DIR="$PWD"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m benchmark dit "$@"
