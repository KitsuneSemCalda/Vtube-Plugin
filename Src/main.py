from Camera.webcam import Webcam
import cv2

cam = Webcam()


def main():
    cam.start()

    while True:
        frame = cam.get_frame()
        cv2.imshow("Webcam 60FPS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    pass


if __name__ == "__main__":
    main()
