import numpy as np
import mediapipe as mp
from collections import deque

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.75
)


class ExpressionDetector:
    def __init__(self, buffer_len: int = 8, speak_min_frames: int = 5):
        self.mouth_buffer = deque(maxlen=buffer_len)
        self.speak_counter = 0
        self.speak_min_frames = speak_min_frames

        self.threshold_open = 0.015
        self.threshold_speak_delta = 0.003

    def discover_expressions(self, frame_rgb: np.ndarray) -> str:
        results = face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return "neutral"

        landmarks = results.multi_face_landmarks[0].landmark
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

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
        if mouth_ratio < 0.015 and smile_ratio > 0.25:
            return "smile"

        if mouth_ratio > self.threshold_open:
            return "open_mouth"

        return "neutral"
