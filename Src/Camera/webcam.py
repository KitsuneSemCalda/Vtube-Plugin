import cv2
from typing import Optional


class Webcam:
    def __init__(
        self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 60
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self):
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open Webcam: {self.device_index}")

    def get_frame(self, to_rgb: bool = True):
        """Return the actual frame from webcam"""
        if self.cap is None:
            raise RuntimeError("Webcam not initialized. Call start() first.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Can't read frame from webcam")
        frame = cv2.resize(frame, (self.width, self.height))
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def release(self):
        """Release the webcam"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
