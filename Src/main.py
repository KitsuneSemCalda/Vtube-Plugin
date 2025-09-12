from Gui.main_window import StartApp

"""

from Camera.webcam import Webcam
from Tracking.face import FaceTracker
from Tracking.discoverExpression import ExpressionDetector
from Output.virtual_cam import VirtualCam


import cv2

cam = Webcam()
ftracker = FaceTracker()
detector = ExpressionDetector()
vcam = VirtualCam()


def main():
    previous_expression = ""
    with cam, vcam:
        while True:
            frame = cam.get_frame()
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
            frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
            frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB)

            landmarks = ftracker.process_frame(frame)
            current_expression = detector.discover_expressions(landmarks)

            if previous_expression != current_expression:
                print(f"The current expression is: {current_expression}")

            previous_expression = current_expression

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Webcam 60FPS", display_frame)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            vcam.send(display_frame)
            vcam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
"""


def main():
    StartApp()


if __name__ == "__main__":
    main()
