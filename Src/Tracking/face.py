import mediapipe as mp
import numpy as np


class FaceTracker:
    def __init__(self, refine_landmarks: bool = True, smoothing: float = 0.9):
        self.refine_landmarks = refine_landmarks
        self.smoothing = smoothing
        self.previous_landmarks = None

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame_rgb: np.ndarray, return_landmarks: bool = False):
        results = self.face_mesh.process(frame_rgb)
        key_data = {"mouth_ratio": 0.0}
        face_landmarks_obj = None

        if results.multi_face_landmarks:
            face_landmarks_obj = results.multi_face_landmarks[0]
            landmarks = face_landmarks_obj.landmark
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

            if self.previous_landmarks is not None:
                landmark_array = (
                    self.smoothing * self.previous_landmarks
                    + (1 - self.smoothing) * landmark_array
                )

            self.previous_landmarks = landmark_array

            key_data["mouth_ratio"] = landmark_array[14, 1] - landmark_array[13, 1]

        if return_landmarks:
            return key_data, face_landmarks_obj
        return key_data

    def get_expression(self, frame_rgb: np.ndarray) -> str:
        data = self.process_frame(frame_rgb)
        mouth_ratio = data.get("mouth_ratio", 0.0)

        return "neutral"

    def close(self):
        self.face_mesh.close()
