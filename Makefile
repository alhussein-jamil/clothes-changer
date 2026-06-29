.PHONY: install install-fast download-models run test lint clean

VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv
# PyTorch CUDA 13 (latest). ORT GPU from PyPI is CUDA 12 — see NVIDIA_CUDA12_LIBS below.
TORCH_INDEX := https://download.pytorch.org/whl/cu130
# ORT GPU links against CUDA 12; PyTorch cu130 only ships cu13 libs (e.g. libcufft.so.12).
NVIDIA_CUDA12_LIBS := \
	nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
	nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12

install:
	$(UV) venv --python 3.10 $(VENV)
	$(UV) pip install -e ".[dev]" --python $(PYTHON)
	@echo ">> Run 'make install-fast' for CUDA torch, ONNX GPU, and model downloads"

install-fast: install
	@echo ">> Installing PyTorch (CUDA 13) + ONNX Runtime GPU (CUDA 12) + NVIDIA libs"
	$(UV) pip install torch torchvision --index-url $(TORCH_INDEX) --python $(PYTHON)
	-$(UV) pip install xformers --index-url $(TORCH_INDEX) --no-deps --python $(PYTHON)
	-$(UV) pip uninstall onnxruntime -y --python $(PYTHON)
	$(UV) pip install onnxruntime-gpu $(NVIDIA_CUDA12_LIBS) gdown --python $(PYTHON)
	$(MAKE) download-models

download-models:
	$(PYTHON) -m clothes_changer.scripts.download_models

run:
	$(PYTHON) -m clothes_changer.main

test:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

lint:
	$(VENV)/bin/ruff check clothes_changer tests

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dist *.egg-info
