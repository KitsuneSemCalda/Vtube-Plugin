import pyvirtualcam
import numpy as np
from typing import Optional
from loguru import logger


class VirtualCam:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
        device: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.cam: Optional[pyvirtualcam.Camera] = None

    def start(self):
        self.cam = pyvirtualcam.Camera(
            width=self.width, height=self.height, fps=self.fps, device=self.device
        )

        logger.trace(
            f"[VirtualCam] VirtualCam initialized: {self.device or 'default'} ({self.width}x{self.height} @ {self.fps}FPS)"
        )

    def sleep_until_next_frame(self):
        if self.cam:
            self.cam.sleep_until_next_frame()

    def send(self, frame: np.ndarray):
        if self.cam is None:
            raise RuntimeError("[VirtualCam]: VirtualCam not founded")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Invalid frame : expected ({self.height},{self.width}), received {frame.shape[:2]}"
            )
        self.cam.send(frame)

    def stop(self):
        if self.cam:
            self.cam.close()
            self.cam = None
            logger.trace("[VirtualCam] virtualcam closed.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
