"""Tests for human parser label mapping."""

import torch

from outfit_studio.ml.parser_labels import masks_from_parser_logits


def test_masks_from_parser_logits_assigns_clothing_and_person():
    logits = torch.zeros(1, 18, 8, 8)
    logits[0, 3, 2:6, 2:6] = 10.0  # top
    logits[0, 1, 0:2, 0:2] = 10.0  # face

    person, clothes = masks_from_parser_logits(logits, confidence=0.35)

    assert person[2:6, 2:6].sum() > 0
    assert clothes[2:6, 2:6].sum() > 0
    assert clothes[0:2, 0:2].sum() == 0


def test_masks_from_parser_logits_excludes_background():
    logits = torch.zeros(1, 18, 4, 4)
    person, clothes = masks_from_parser_logits(logits, confidence=0.35)
    assert person.sum() == 0
    assert clothes.sum() == 0
