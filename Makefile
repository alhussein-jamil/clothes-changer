# Outfit Studio — local + Docker shortcuts
#
#   make              # help
#   make install-fast # venv + models
#   make run          # Gradio UI
#   make stop         # stop local demo

ROOT        := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV          := uv
PYTHON      := $(ROOT)/.venv/bin/python
COMPOSE     := docker compose
COMPOSE_GPU := $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml

# ── colors (auto-disable if not a TTY / NO_COLOR set) ─────────────────
ifeq ($(NO_COLOR),)
  HAS_TTY := $(shell [ -t 1 ] && echo 1)
endif
ifeq ($(HAS_TTY),1)
  C_RESET   := \033[0m
  C_BOLD    := \033[1m
  C_DIM     := \033[2m
  C_RED     := \033[31m
  C_GREEN   := \033[32m
  C_YELLOW  := \033[33m
  C_CYAN    := \033[36m
  C_PINK    := \033[38;5;213m
  C_ORANGE  := \033[38;5;214m
  C_TEAL    := \033[38;5;44m
else
  C_RESET :=
  C_BOLD :=
  C_DIM :=
  C_RED :=
  C_GREEN :=
  C_YELLOW :=
  C_CYAN :=
  C_PINK :=
  C_ORANGE :=
  C_TEAL :=
endif

define banner
	@printf '$(C_PINK)$(C_BOLD)\n'
	@printf '  ╔══════════════════════════════════════╗\n'
	@printf '  ║       Outfit Studio  ·  make        ║\n'
	@printf '  ╚══════════════════════════════════════╝\n'
	@printf '$(C_RESET)\n'
endef

define say
	@printf '$(C_TEAL)$(C_BOLD)➜$(C_RESET) $(C_BOLD)%s$(C_RESET)\n' "$(1)"
endef

define ok
	@printf '$(C_GREEN)✓$(C_RESET) %s\n' "$(1)"
endef

define warn
	@printf '$(C_YELLOW)!$(C_RESET) %s\n' "$(1)"
endef

.DEFAULT_GOAL := help

.PHONY: help install install-fast download-models fix-ort-gpu \
	run stop test lint clean add-user \
	docker-build docker-up docker-up-cpu docker-down docker-logs docker-download-models

# ── help ──────────────────────────────────────────────────────────────
help:
	$(banner)
	@printf '$(C_DIM)  UI  http://localhost:7860$(C_RESET)\n\n'
	@printf '$(C_ORANGE)$(C_BOLD)  Setup$(C_RESET)\n'
	@printf '    $(C_CYAN)install$(C_RESET)                 sync .venv (incl. dev extras)\n'
	@printf '    $(C_CYAN)install-fast$(C_RESET)            install + download models\n'
	@printf '    $(C_CYAN)download-models$(C_RESET)         fetch checkpoints / HF weights\n\n'
	@printf '$(C_ORANGE)$(C_BOLD)  Run$(C_RESET)\n'
	@printf '    $(C_CYAN)run$(C_RESET)                     start Gradio demo (foreground)\n'
	@printf '    $(C_CYAN)stop$(C_RESET)                    stop local demo\n'
	@printf '    $(C_CYAN)add-user$(C_RESET)                USER= PASS= [CREDITS=] [ADMIN=true]\n\n'
	@printf '$(C_ORANGE)$(C_BOLD)  Quality$(C_RESET)\n'
	@printf '    $(C_CYAN)test$(C_RESET)                    pytest (skip slow)\n'
	@printf '    $(C_CYAN)lint$(C_RESET)                    ruff check\n'
	@printf '    $(C_CYAN)clean$(C_RESET)                   wipe .venv + caches\n\n'
	@printf '$(C_DIM)  TensorTorrent streams oversized UNets beyond VRAM$(C_RESET)\n'
	@printf '$(C_DIM)  (OUTFIT_STUDIO_TENSOR_TORRENT=true; cache .cache/tensortorrent)$(C_RESET)\n\n'
	@printf '$(C_ORANGE)$(C_BOLD)  Docker$(C_RESET)\n'
	@printf '    $(C_CYAN)docker-build$(C_RESET)            build image\n'
	@printf '    $(C_CYAN)docker-up$(C_RESET)               up (GPU compose)\n'
	@printf '    $(C_CYAN)docker-up-cpu$(C_RESET)           up (CPU only)\n'
	@printf '    $(C_CYAN)docker-down$(C_RESET)             stop containers\n'
	@printf '    $(C_CYAN)docker-logs$(C_RESET)             follow app logs\n'
	@printf '    $(C_CYAN)docker-download-models$(C_RESET)  download models in container\n\n'
	@printf '$(C_DIM)  tip: NO_COLOR=1 make …  disables ANSI$(C_RESET)\n\n'

# ── setup ─────────────────────────────────────────────────────────────
fix-ort-gpu:
	@if [ "$$(uname -s)" = "Linux" ]; then PYTHON=$(PYTHON) ./docker/fix-ort-gpu.sh; fi

install:
	$(call say,install → uv sync --extra dev)
	$(UV) sync --frozen --extra dev
	@$(MAKE) --no-print-directory fix-ort-gpu
	$(call ok,venv ready)

install-fast: install
	@$(MAKE) --no-print-directory download-models

download-models:
	$(call say,download models)
	$(UV) run outfit-studio-download-models
	$(call ok,models ready)

# ── run ───────────────────────────────────────────────────────────────
run:
	$(call say,run outfit-studio)
	$(UV) run outfit-studio

stop:
	$(call say,stop local demo)
	@found=0; \
	for pid in $$(pgrep -f '$(ROOT)/[.]venv/bin/outfit-studio' 2>/dev/null || true); do \
		ppid=$$(ps -o ppid= -p $$pid 2>/dev/null | tr -d ' '); \
		printf '$(C_DIM)  kill outfit-studio pid %s$(C_RESET)\n' "$$pid"; \
		kill $$pid 2>/dev/null || true; \
		if [ -n "$$ppid" ] && [ "$$ppid" -gt 1 ]; then \
			cmd=$$(ps -o args= -p $$ppid 2>/dev/null || true); \
			case "$$cmd" in \
				*uv*"outfit-studio"*) \
					printf '$(C_DIM)  kill uv parent pid %s$(C_RESET)\n' "$$ppid"; \
					kill $$ppid 2>/dev/null || true; \
					;; \
			esac; \
		fi; \
		found=1; \
	done; \
	if [ "$$found" -eq 0 ]; then \
		printf '$(C_YELLOW)!$(C_RESET) no local outfit-studio demo running\n'; \
	else \
		sleep 0.4; \
		pkill -9 -f '$(ROOT)/[.]venv/bin/outfit-studio' 2>/dev/null || true; \
		printf '$(C_GREEN)✓$(C_RESET) stopped\n'; \
	fi

add-user:
ifndef USER
	$(error Usage: make add-user USER=name PASS=password [CREDITS=10] [ADMIN=true])
endif
ifndef PASS
	$(error Usage: make add-user USER=name PASS=password [CREDITS=10] [ADMIN=true])
endif
	$(call say,add-user $(USER))
	$(UV) run outfit-studio-add-user $(USER) $(PASS) \
		$(if $(CREDITS),--credits $(CREDITS),) \
		$(if $(filter true yes 1,$(ADMIN)),--admin,)
	$(call ok,user $(USER) ready)

# ── quality ───────────────────────────────────────────────────────────
test:
	$(call say,test)
	$(UV) run pytest tests/ -v -m "not slow"

lint:
	$(call say,lint)
	$(UV) run ruff check outfit_studio tests
	$(call ok,ruff clean)

clean:
	$(call say,clean venv + caches)
	rm -rf .venv .pytest_cache .ruff_cache dist *.egg-info
	$(call ok,cleaned)

# ── docker ────────────────────────────────────────────────────────────
docker-build:
	$(call say,docker build)
	$(COMPOSE) build
	$(call ok,image built)

docker-up:
	$(call say,docker up \(GPU\))
	$(COMPOSE_GPU) up -d
	$(call ok,containers up)

docker-up-cpu:
	$(call say,docker up \(CPU\))
	$(COMPOSE) up -d
	$(call ok,containers up)

docker-down:
	$(call say,docker down)
	$(COMPOSE) down
	$(call ok,containers stopped)

docker-logs:
	$(call say,docker logs)
	$(COMPOSE) logs -f outfit-studio

docker-download-models:
	$(call say,docker download models)
	$(COMPOSE) exec outfit-studio outfit-studio-download-models
	$(call ok,models ready in container)
