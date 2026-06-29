# Clothes Changer

AI outfit inpainting: upload a photo, auto-segment clothing, edit the mask, and inpaint a new outfit with Stable Diffusion.

## Requirements

- Python 3.10–3.12
- [uv](https://github.com/astral-sh/uv)
- NVIDIA GPU recommended (8 GB+ VRAM for SD1.5 inpainting)

## Quick start

```bash
cd clothes-changer
cp .env.example .env
make install-fast    # CUDA torch, ONNX GPU, downloads segmentation + default inpaint model
make run             # http://localhost:7860
```

Default login: `admin` / `admin`

## Make targets

| Target | Description |
|--------|-------------|
| `make install` | Create venv and install Python deps |
| `make install-fast` | Above + CUDA PyTorch, ONNX GPU, model download |
| `make download-models` | Segmentation weights + default inpaint checkpoint |
| `make run` | Start the Gradio UI |
| `make test` | Run unit tests |
| `make lint` | Ruff check |

## Configuration

### Runtime (`.env`)

Deployment settings only — host, ports, paths, auth, debug flags. See `.env.example`.

| Setting | Purpose |
|---------|---------|
| `CLOTHES_CHANGER_HOST` / `PORT` | Server bind address |
| `CLOTHES_CHANGER_MODELS_DIR` / `OUTPUT_DIR` / `DB_PATH` | Local paths |
| `CLOTHES_CHANGER_DEFAULT_*` / `REQUIRE_AUTH` | Login and credits |
| `CLOTHES_CHANGER_DEBUG` / `LOG_LEVEL` / `PIPELINE_DEBUG*` | Diagnostics |

Models, prompts, and generation defaults are **not** in `.env`.

### Content / ML (`config/`)

Branding, prompts, checkpoints, and generation tuning live in YAML:

| File | Tracked | Purpose |
|------|---------|---------|
| `config/content.default.yaml` | Yes | Safe vanilla defaults shipped with the repo |
| `config/content.local.yaml` | No (gitignored) | Local overrides — prompts, default checkpoint, download URLs |
| `config/content.local.yaml.example` | Yes | Template for local overrides |

```bash
cp config/content.local.yaml.example config/content.local.yaml
# edit config/content.local.yaml — models.default_inpaint, prompts, generation, etc.
```

Local YAML merges on top of the default file.

**Shipped default checkpoint:** `runwayml/stable-diffusion-inpainting` (Hugging Face).

**Local example** sets `cyberrealistic_v80Inpainting.safetensors` with Civitai download URLs. After copying, `make download-models` fetches the configured default into `./models/`.

## Stack

| Component | Technology |
|-----------|-------------|
| UI | Gradio 4 + ImageSlider |
| Segmentation | SegFormer B2 + U2NET |
| Pose (optional) | rtmlib ONNX + ControlNet OpenPose |
| Inpainting | Diffusers SD1.5 inpaint |

## Development

```bash
make lint
make test
pre-commit install
pre-commit run --all-files
```

## CLI

```bash
.venv/bin/clothes-changer-download-models
.venv/bin/clothes-changer
```
