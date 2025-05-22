# drowsiness_detection.py

import cv2
import dlib
import numpy as np
from imutils import face_utils
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])   # 49, 55
    return (A + B) / (2.0 * C)

def run_drowsiness_detection():
    EAR_THRESHOLD = 0.25
    MAR_THRESHOLD = 0.75
    FPS = 30
    TIRED_SECONDS = 2.0
    CONSEC_FRAMES = int(FPS * TIRED_SECONDS)
    COUNTER = 0
    ALARM_ON = False
    ALARM_END_TIME = 0
    ALERT_DURATION = 3

    #predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("[錯誤] 找不到 Dlib 特徵點模型，請先下載：")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    print("[INFO] 載入人臉偵測與特徵點模型...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255,0,0), 1)

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        ALARM_END_TIME = time.time() + ALERT_DURATION
            else:
                COUNTER = 0
                if time.time() >= ALARM_END_TIME:
                    ALARM_ON = False

            if ALARM_ON:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Drowsiness and Yawning Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_drowsiness_detection()
