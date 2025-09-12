from typing import Optional
from collections import deque
import numpy as np


class ExpressionDetector:
    def __init__(self):
        self.history_length: int = 5
        self.mouth_open_thresh = 0.015
        self.smile_thresh = 0.25
        self.history = deque(maxlen=self.history_length)

    def _compute_mouth_metrics(self, landmarks: np.ndarray) -> tuple[float, float]:
        top = landmarks[10, 1]
        bottom = landmarks[152, 1]
        face_height = max(bottom - top, 0.1)

        upper_lip = np.mean(landmarks[[13, 312, 317], 1])
        lower_lip = np.mean(landmarks[[14, 82, 87], 1])
        mouth_ratio = (lower_lip - upper_lip) / face_height

        left_corner = landmarks[61, :2]
        right_corner = landmarks[291, :2]
        mouth_width = np.linalg.norm(left_corner - right_corner)
        smile_ratio = mouth_width / face_height

        return mouth_ratio, smile_ratio

    def _compute_additional_metrics(self, landmarks: np.ndarray) -> dict:
        eyebrow_distance = np.linalg.norm(landmarks[105, :2] - landmarks[334, :2])
        return {"eyebrow_distance": eyebrow_distance}

    def _classify_expression(
        self, mouth_ratio: float, smile_ratio: float, metrics: dict
    ) -> str:
        if mouth_ratio < self.mouth_open_thresh and smile_ratio > self.smile_thresh:
            return "smile"
        if mouth_ratio > self.mouth_open_thresh:
            return "open_mouth"
        return "neutral"

    def discover_expressions(self, landmarks: Optional[np.ndarray]) -> str:
        if landmarks is None or np.isnan(landmarks).any():
            expression = "neutral"
        else:
            mouth_ratio, smile_ratio = self._compute_mouth_metrics(landmarks)
            metrics = self._compute_additional_metrics(landmarks)
            expression = self._classify_expression(mouth_ratio, smile_ratio, metrics)

        self.history.append(expression)
        return max(set(self.history), key=self.history.count)
