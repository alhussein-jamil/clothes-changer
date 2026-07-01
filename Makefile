.PHONY: install install-fast download-models run test lint clean add-user fix-ort-gpu \
	docker-build docker-up docker-up-cpu docker-down docker-logs docker-download-models

UV := uv
PYTHON := .venv/bin/python
COMPOSE := docker compose
COMPOSE_GPU := $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml

fix-ort-gpu:
	@if [ "$$(uname -s)" = "Linux" ]; then PYTHON=$(PYTHON) ./docker/fix-ort-gpu.sh; fi

install:
	$(UV) sync --frozen --extra dev
	@$(MAKE) --no-print-directory fix-ort-gpu

install-fast: install
	$(MAKE) download-models

download-models:
	$(UV) run outfit-studio-download-models

run:
	$(UV) run outfit-studio

test:
	$(UV) run pytest tests/ -v -m "not slow"

lint:
	$(UV) run ruff check outfit_studio tests

add-user:
ifndef USER
	$(error Usage: make add-user USER=name PASS=password [CREDITS=10] [ADMIN=true])
endif
ifndef PASS
	$(error Usage: make add-user USER=name PASS=password [CREDITS=10] [ADMIN=true])
endif
	$(UV) run outfit-studio-add-user $(USER) $(PASS) \
		$(if $(CREDITS),--credits $(CREDITS),) \
		$(if $(filter true yes 1,$(ADMIN)),--admin,)

clean:
	rm -rf .venv .pytest_cache .ruff_cache dist *.egg-info

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE_GPU) up -d

docker-up-cpu:
	$(COMPOSE) up -d

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f outfit-studio

docker-download-models:
	$(COMPOSE) exec outfit-studio outfit-studio-download-models
