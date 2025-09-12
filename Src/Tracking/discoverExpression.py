from typing import Optional
import numpy as np


class ExpressionDetector:
    def __init__(self, threshold_open: float = 0.015, threshold_smile: float = 0.25):
        self.threshold_open = threshold_open
        self.threshold_smile = threshold_smile

    def discover_expressions(self, landmarks: Optional[np.ndarray]) -> str:
        if landmarks is None:
            return "neutral"

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

        if mouth_ratio < self.threshold_open and smile_ratio > self.threshold_smile:
            return "smile"
        if mouth_ratio > self.threshold_open:
            return "open_mouth"
        return "neutral"
