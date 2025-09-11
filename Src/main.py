from Camera.webcam import Webcam
from Tracking.face import FaceTracker
from Output.virtual_cam import VirtualCam

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cam = Webcam()
ftracker = FaceTracker()
vcam = VirtualCam()


def main():
    with cam, vcam:
        while True:
            frame = cam.get_frame()

            # Pré-processamento
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
            frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
            frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB)

            # Tracking facial otimizado
            landmarks_data, face_landmarks = ftracker.process_frame(
                frame, return_landmarks=True
            )

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Desenhar landmarks apenas se houver face
            if face_landmarks:
                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            # Exibição local
            cv2.imshow("Webcam 60FPS", display_frame)

            # Envio para VirtualCam
            vcam.send(display_frame)
            vcam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

