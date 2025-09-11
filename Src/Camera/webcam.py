import cv2
from typing import Optional


class Webcam:
    def __init__(
        self, device_index: int = 0, width: int = 640, height: int = 800, fps: int = 60
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open Webcam: {self.device_index}")

    def get_frame(self):
        """Return the actual frame from webcam"""
        if self.cap is None:
            raise RuntimeError("Webcam can't initialized")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Can't open frame of webcam")
        return frame

    def release(self):
        """Release the webcam"""
        if self.cap:
            self.cap.release()
            self.cap = None
