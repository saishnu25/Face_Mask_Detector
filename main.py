import cv2
import numpy as np
import os

CASCADE_PATH = r"C:\Users\chuen\miniconda3\envs\Arduino2Python\Lib\site-packages\cv2\data"

face_haarcascade = os.path.join(CASCADE_PATH, "haarcascade_frontalface_default.xml")
eye_haarcascade = os.path.join(CASCADE_PATH, "haarcascade_eye.xml")

face_detector = cv2.CascadeClassifier(face_haarcascade)
eye_detector = cv2.CascadeClassifier(eye_haarcascade)

camera = cv2.VideoCapture(0)

def main():
    
    while True:
        _, frame = camera.read()
        print(frame.shape)

        print(os.getcwd())
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame_gray, 1.3, 5, minSize=(120, 120))
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Obtain the eyes now and use the facial detection region as our ROI (Region of Interest)
            roi_gray = frame_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))

            eyes = eye_detector.detectMultiScale(roi_gray, 1.03, 5, minSize=(40, 40))
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)

        cv2.imshow("Frame OG", frame)
        cv2.imshow("Frame GRAY", frame_gray)

        # Press Key Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return 


if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()