import mediapipe as mp
import numpy as np
from typing import Optional
from Tracking.discoverExpression import ExpressionDetector


class FaceTracker:
    def __init__(self, refine_landmarks: bool = True, smoothing: float = 0.9):
        self.refine_landmarks = refine_landmarks
        self.smoothing = smoothing
        self.previous_landmarks: Optional[np.ndarray] = None

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        if self.previous_landmarks is not None:
            landmark_array = (
                self.smoothing * self.previous_landmarks
                + (1 - self.smoothing) * landmark_array
            )

        self.previous_landmarks = landmark_array
        return landmark_array

    def close(self):
        self.face_mesh.close()
