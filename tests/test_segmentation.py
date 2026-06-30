"""ML integration tests (download models on first run)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.slow


@patch("outfit_studio.ml.segmentor.SegformerImageProcessor")
@patch("outfit_studio.ml.segmentor.AutoModelForSemanticSegmentation")
def test_clothes_segmentor_runs(mock_auto_model, mock_processor):
    from outfit_studio.ml.segmentor import ClothesSegmentor

    img = Image.new("RGB", (64, 64), color=(180, 140, 120))
    h, w = 64, 64

    mock_proc = MagicMock()
    mock_processor.from_pretrained.return_value = mock_proc
    batch = MagicMock()
    batch.to.return_value = {"pixel_values": torch.zeros(1, 3, 64, 64)}
    mock_proc.return_value = batch

    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model
    logits = torch.zeros(1, 18, 16, 16)
    logits[0, 4, 4:10, 4:10] = 10.0  # clothes category
    logits[0, 1, 2:8, 2:8] = 10.0  # person category
    model_out = MagicMock()
    model_out.logits = logits
    mock_model.return_value = model_out

    person, clothes = ClothesSegmentor().segment(img)

    assert person.shape == (h, w)
    assert clothes.shape == (h, w)
    assert person.sum() > 0
