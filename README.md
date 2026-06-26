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
make install-fast    # CUDA torch, ONNX GPU, downloads default inpaint model
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

Settings use the `CLOTHES_CHANGER_` prefix — see `.env.example`.

Models and outputs are stored under `./models` and `./outputs` by default.

### Content / prompts (`config/`)

Branding, prompts, and downloadable model URLs live in YAML:

| File | Tracked | Purpose |
|------|---------|---------|
| `config/content.default.yaml` | Yes | Safe defaults shipped with the repo |
| `config/content.local.yaml` | No (gitignored) | Local overrides for custom prompts/branding |
| `config/content.local.yaml.example` | Yes | Template for local overrides |

```bash
cp config/content.local.yaml.example config/content.local.yaml
# edit config/content.local.yaml — prompts, app name, etc.
```

Local YAML merges on top of the default file.

## Stack

| Component | Technology |
|-----------|-------------|
| UI | Gradio 5 + ImageSlider |
| Segmentation | SegFormer B2 + U2NET |
| Pose (optional) | rtmlib ONNX + ControlNet OpenPose |
| Inpainting | Diffusers SD1.5 inpaint |

Default inpaint model: **Realistic Vision v6 inpaint** (~2 GB, downloaded on `make install-fast`).

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
