"""
MediaPipe Hands wrapper. Same detection path for training and API.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np

from app.services.feature_extractor import hand_landmarks_to_features


class MediaPipeHandService:
    def __init__(self) -> None:
        import mediapipe as mp

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    def landmarks_from_rgb(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be (H, W, 3)")
        rgb_u8 = np.ascontiguousarray(rgb)
        results = self._hands.process(rgb_u8)
        if not results.multi_hand_landmarks:
            results = self._hands.process(rgb_u8)
        if not results.multi_hand_landmarks:
            return None
        lm = results.multi_hand_landmarks[0].landmark
        out = np.empty((21, 3), dtype=np.float32)
        for i, p in enumerate(lm):
            out[i, 0] = p.x
            out[i, 1] = p.y
            out[i, 2] = p.z
        return out

    def feature_vector_from_bgr(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        coords = self.landmarks_from_rgb(rgb)
        if coords is None:
            return None
        return hand_landmarks_to_features(coords)
