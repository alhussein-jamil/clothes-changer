import numpy as np

from outfit_studio.utils.hand_mask import (
    OPENPOSE_LEFT_HAND,
    OPENPOSE_RIGHT_HAND,
    build_combined_hand_mask,
    hand_regions_from_pose,
    subtract_hand_mask,
)


def _synthetic_pose(width: int = 400, height: int = 600) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((134, 2), dtype=np.float32)
    scores = np.zeros(134, dtype=np.float32)

    def place(indices: tuple[int, ...], cx: float, cy: float, spread: float = 30.0) -> None:
        for i, idx in enumerate(indices):
            keypoints[idx] = (cx + (i % 5) * spread / 5, cy + (i // 5) * spread / 4)
            scores[idx] = 0.95

    place(OPENPOSE_RIGHT_HAND, 120, 420)
    place(OPENPOSE_LEFT_HAND, 280, 420)
    return keypoints, scores


def test_hand_regions_from_pose_finds_both_hands():
    keypoints, scores = _synthetic_pose()
    regions = hand_regions_from_pose(
        keypoints,
        scores,
        kpt_thr=0.3,
        image_size=(400, 600),
    )
    assert len(regions) == 2


def test_hand_regions_skip_low_confidence():
    keypoints, scores = _synthetic_pose()
    scores[list(OPENPOSE_LEFT_HAND)] = 0.0
    regions = hand_regions_from_pose(keypoints, scores, kpt_thr=0.3, image_size=(400, 600))
    assert len(regions) == 1


def test_build_combined_hand_mask_covers_hands():
    keypoints, scores = _synthetic_pose()
    mask = build_combined_hand_mask(keypoints, scores, (600, 400), kpt_thr=0.3)
    assert mask[420, 120] > 0
    assert mask[420, 280] > 0
    assert mask[100, 200] == 0


def test_subtract_hand_mask_removes_overlap():
    clothes = np.zeros((100, 100), dtype=np.uint8)
    clothes[40:80, 20:80] = 1
    hand = np.zeros((100, 100), dtype=np.uint8)
    hand[50:70, 30:50] = 255
    result = subtract_hand_mask(clothes, hand)
    assert result[55, 40] == 0
    assert result[45, 70] == 1
