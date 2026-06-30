"""Human parser label groups and mask extraction."""

from __future__ import annotations

import torch

from outfit_studio.constants import CLOTHES_PARSER_CATEGORIES, PERSON_PARSER_CATEGORIES

_CLOTHES_CATEGORY_TENSORS: dict[torch.device, torch.Tensor] = {}
_PERSON_CATEGORY_TENSORS: dict[torch.device, torch.Tensor] = {}


def _clothes_categories(device: torch.device) -> torch.Tensor:
    if device not in _CLOTHES_CATEGORY_TENSORS:
        _CLOTHES_CATEGORY_TENSORS[device] = torch.tensor(CLOTHES_PARSER_CATEGORIES, device=device)
    return _CLOTHES_CATEGORY_TENSORS[device]


def _person_categories(device: torch.device) -> torch.Tensor:
    if device not in _PERSON_CATEGORY_TENSORS:
        _PERSON_CATEGORY_TENSORS[device] = torch.tensor(PERSON_PARSER_CATEGORIES, device=device)
    return _PERSON_CATEGORY_TENSORS[device]


def masks_from_parser_logits(
    logits: torch.Tensor,
    *,
    confidence: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build person and clothing masks from parser logits."""
    device = logits.device
    pred_seg = logits.argmax(dim=1)[0]
    clothes_idx = _clothes_categories(device)
    person_idx = _person_categories(device)

    # Group softmax probabilities without materializing a full (C,H,W) tensor.
    log_denom = torch.logsumexp(logits, dim=1)
    clothes_log = torch.logsumexp(logits.index_select(1, clothes_idx), dim=1)
    person_log = torch.logsumexp(logits.index_select(1, person_idx), dim=1)
    clothes_prob = (clothes_log - log_denom)[0]
    person_prob = (person_log - log_denom)[0]

    person_mask = (pred_seg != 0).float()
    clothes_mask = torch.isin(pred_seg, clothes_idx).float()

    confident_clothes = (clothes_prob > confidence) & (clothes_prob > person_prob)
    clothes_mask = torch.logical_or(clothes_mask > 0, confident_clothes).float()

    return person_mask, clothes_mask
