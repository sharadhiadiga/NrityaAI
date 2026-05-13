"""
MediaPipe (21,3) landmarks -> fixed-length feature vector (< 120).
Identical pipeline for train_model.py and mediapipe_service (inference).
"""

from __future__ import annotations

import itertools
import numpy as np

_FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)
_MCP_FROM_WRIST: tuple[int, ...] = (2, 5, 9, 13, 17)  # thumb MCP + four fingers MCP

_ANGLE_PIP: tuple[tuple[int, int, int], ...] = (
    (2, 3, 4),
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)
_ANGLE_DIP: tuple[tuple[int, int, int], ...] = (
    (6, 7, 8),
    (10, 11, 12),
    (14, 15, 16),
    (18, 19, 20),
)
_ANGLE_MCP: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (0, 5, 6),
    (0, 9, 10),
    (0, 13, 14),
    (0, 17, 18),
)

_FT_PAIRS: list[tuple[int, int]] = []
for i, a in enumerate(_FINGERTIPS):
    for b in _FINGERTIPS[i + 1 :]:
        _FT_PAIRS.append((a, b))

_N_MCP_OPEN = len(list(itertools.combinations(range(len(_MCP_FROM_WRIST)), 2)))

FEATURE_VECTOR_SIZE: int = (
    20 * 3
    + len(_FINGERTIPS)
    + len(_FT_PAIRS)
    + len(_ANGLE_PIP)
    + len(_ANGLE_DIP)
    + len(_ANGLE_MCP)
    + _N_MCP_OPEN
)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))


def _angle_at_vertex(rel: np.ndarray, ia: int, iv: int, ic: int) -> float:
    v1 = rel[ia] - rel[iv]
    v2 = rel[ic] - rel[iv]
    n1 = float(np.sqrt(np.dot(v1, v1)))
    n2 = float(np.sqrt(np.dot(v2, v2)))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.arccos(c))


def _angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    n1 = float(np.sqrt(np.dot(a, a)))
    n2 = float(np.sqrt(np.dot(b, b)))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    c = float(np.dot(a, b) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.arccos(c))


def hand_landmarks_to_features(coords: np.ndarray) -> np.ndarray:
    """
    Wrist-relative coords, scaled by mean wrist–fingertip distance, plus geometry.

    - 60: scaled (x,y,z) for joints 1..20
    - 5: scaled wrist→fingertip distances
    - 10: scaled fingertip–fingertip distances
    - 5 + 4 + 5: joint angles (PIP, DIP, MCP), radians
    - 10: pairwise angles between wrist→MCP vectors (hand opening)
    Total FEATURE_VECTOR_SIZE (99 < 120).
    """
    if coords.shape != (21, 3):
        raise ValueError(f"Expected coords shape (21, 3), got {coords.shape}")
    rel = (coords - coords[0]).astype(np.float32, copy=False)

    tip_lens = np.array([_l2(rel[0], rel[i]) for i in _FINGERTIPS], dtype=np.float64)
    scale = float(np.mean(tip_lens))
    scale = max(scale, 1e-6)

    pos = (rel[1:] / scale).reshape(-1).astype(np.float32, copy=False)
    wrist_tip = (tip_lens / scale).astype(np.float32, copy=False)
    ft_dists = np.array(
        [_l2(rel[i], rel[j]) / scale for i, j in _FT_PAIRS], dtype=np.float32
    )

    ang_pip = np.array(
        [_angle_at_vertex(rel, a, v, c) for a, v, c in _ANGLE_PIP], dtype=np.float32
    )
    ang_dip = np.array(
        [_angle_at_vertex(rel, a, v, c) for a, v, c in _ANGLE_DIP], dtype=np.float32
    )
    ang_mcp = np.array(
        [_angle_at_vertex(rel, a, v, c) for a, v, c in _ANGLE_MCP], dtype=np.float32
    )

    mcp_vecs = [rel[i].astype(np.float64, copy=False) for i in _MCP_FROM_WRIST]
    mcp_open = []
    for i, j in itertools.combinations(range(len(mcp_vecs)), 2):
        mcp_open.append(_angle_between_vectors(mcp_vecs[i], mcp_vecs[j]))
    mcp_open_arr = np.array(mcp_open, dtype=np.float32)

    out = np.concatenate([pos, wrist_tip, ft_dists, ang_pip, ang_dip, ang_mcp, mcp_open_arr])
    assert out.shape[0] == FEATURE_VECTOR_SIZE
    return out.astype(np.float32, copy=False)
