from Camera.webcam import Webcam
import cv2

cam = Webcam()

while True:
    frame = cam.get_frame()
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
