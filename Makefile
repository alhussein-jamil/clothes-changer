.PHONY: install install-fast download-models run test lint clean add-user

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
	@if [ "$$(uname -s)" = "Linux" ]; then \
		$(UV) pip uninstall onnxruntime -y --python $(PYTHON) 2>/dev/null || true; \
		$(UV) pip install --reinstall onnxruntime-gpu $(NVIDIA_CUDA12_LIBS) --python $(PYTHON); \
	fi
	@echo ">> Run 'make install-fast' for CUDA torch and model downloads"

install-fast: install
	@echo ">> Installing PyTorch (CUDA 13)"
	$(UV) pip install torch torchvision --index-url $(TORCH_INDEX) --python $(PYTHON)
	-$(UV) pip install xformers --index-url $(TORCH_INDEX) --no-deps --python $(PYTHON)
	$(MAKE) download-models

download-models:
	$(PYTHON) -m outfit_studio.scripts.download_models

run:
	$(PYTHON) -m outfit_studio.main

test:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

lint:
	$(VENV)/bin/ruff check outfit_studio tests

add-user:
ifndef USER
	$(error Usage: make add-user USER=name PASS=password [CREDITS=10] [ADMIN=true])
endif
ifndef PASS
	$(error Usage: make add-user USER=name PASS=password [CREDITS=10] [ADMIN=true])
endif
	$(PYTHON) -m outfit_studio.scripts.add_user $(USER) $(PASS) \
		$(if $(CREDITS),--credits $(CREDITS),) \
		$(if $(filter true yes 1,$(ADMIN)),--admin,)

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dist *.egg-info
