#!/usr/bin/env sh
# rtmlib pulls CPU onnxruntime; swap to GPU build on Linux.
set -eu

PYTHON="${PYTHON:-.venv/bin/python}"
uv pip uninstall onnxruntime -y 2>/dev/null || true
uv pip install --reinstall-package onnxruntime-gpu onnxruntime-gpu --python "$PYTHON"
