"""U2NET cloth segmentation."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from outfit_studio.constants import (
    U2NET_INPUT_CHANNELS,
    U2NET_INPUT_SIZE,
    U2NET_NORMALIZE_MEAN,
    U2NET_NORMALIZE_STD,
    U2NET_OUTPUT_CLASSES,
    U2NET_TENSOR_CHANNELS,
)
from outfit_studio.ml.network import U2NET

logger = logging.getLogger(__name__)

SHAPE1, SHAPE3, SHAPE18 = U2NET_TENSOR_CHANNELS


def load_checkpoint(model: U2NET, checkpoint_path: str | Path) -> U2NET:
    path = Path(checkpoint_path)
    if not path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise FileNotFoundError(msg)
    logger.info("Loading U2NET checkpoint from %s", path)
    # cloth_segm.pth predates PyTorch 2.6 safe unpickling (weights_only=True).
    model_state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logger.debug("U2NET checkpoint loaded (%d tensors)", len(new_state_dict))
    return model


def get_palette(num_cls: int) -> list[int]:
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


class NormalizeImage:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std
        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * SHAPE3, [self.std] * SHAPE3)
        self.normalize_18 = transforms.Normalize([self.mean] * SHAPE18, [self.std] * SHAPE18)

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.shape[0] == SHAPE1:
            return self.normalize_1(image_tensor)
        if image_tensor.shape[0] == SHAPE3:
            return self.normalize_3(image_tensor)
        if image_tensor.shape[0] == SHAPE18:
            return self.normalize_18(image_tensor)
        return image_tensor


def apply_transform(img: Image.Image) -> torch.Tensor:
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(), NormalizeImage(U2NET_NORMALIZE_MEAN, U2NET_NORMALIZE_STD)]
    )
    return transform_rgb(img)


def generate_mask(
    input_image: Image.Image,
    net: U2NET,
    palette: list[int],
    device: str = "cpu",
) -> Image.Image:
    img_size = input_image.size
    logger.debug(
        "U2NET inference %dx%d → %dx%d on %s",
        *img_size,
        U2NET_INPUT_SIZE,
        U2NET_INPUT_SIZE,
        device,
    )
    img = input_image.resize((U2NET_INPUT_SIZE, U2NET_INPUT_SIZE), Image.BICUBIC)
    image_tensor = torch.unsqueeze(apply_transform(img), 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = torch.nn.functional.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode="P")
    cloth_seg.putpalette(palette)
    return cloth_seg.resize(img_size, Image.BICUBIC)


def load_seg_model(checkpoint_path: str | Path, device: str = "cpu") -> U2NET:
    """Load U2NET on CPU first — safe when diffusers loads concurrently (meta tensors)."""
    logger.debug("Building U2NET on device=%s", device)
    path = Path(checkpoint_path)
    raw_state = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = OrderedDict()
    for k, v in raw_state.items():
        name = k[7:] if k.startswith("module.") else k
        state_dict[name] = v

    with torch.device("cpu"):
        net = U2NET(in_ch=U2NET_INPUT_CHANNELS, out_ch=U2NET_OUTPUT_CLASSES)
        net.load_state_dict(state_dict)
        net.eval()

    if device != "cpu":
        net = net.to(device)
    logger.debug("U2NET ready on %s (%d tensors)", device, len(state_dict))
    return net
