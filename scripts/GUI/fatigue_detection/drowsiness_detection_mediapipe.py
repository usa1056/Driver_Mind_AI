import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk

mp_face_mesh = mp.solutions.face_mesh

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(eye_landmarks):
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_landmarks):
    A = euclidean_distance(mouth_landmarks[13], mouth_landmarks[11])
    B = euclidean_distance(mouth_landmarks[9], mouth_landmarks[7])
    C = euclidean_distance(mouth_landmarks[0], mouth_landmarks[10])
    return (A + B) / (2.0 * C)

def run_drowsiness_detection():
    FPS = 30
    TIRED_SECONDS = 2.0
    CONSEC_FRAMES = int(FPS * TIRED_SECONDS)
    COUNTER = 0
    ALARM_ON = False
    ALARM_END_TIME = 0
    ALERT_DURATION = 3

    EAR_CALIBRATION_TIME = 3  # 校正時間（秒）
    CALIBRATION_FRAMES = int(FPS * EAR_CALIBRATION_TIME)
    calibration_ears = []
    EAR_THRESHOLD = None  # 尚未初始化

    MAR_THRESHOLD = 0.75

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_count = 0
        calibration_done = False
        calibration_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                mouth_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311]

                left_eye = [landmarks[i] for i in left_eye_indices]
                right_eye = [landmarks[i] for i in right_eye_indices]
                mouth = [landmarks[i] for i in mouth_indices]

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)

                if not calibration_done:
                    calibration_ears.append(ear)
                    frame_count += 1
                    remaining_time = EAR_CALIBRATION_TIME - int(time.time() - calibration_start)
                    cv2.putText(frame, f"Calibrating... {remaining_time}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    if frame_count >= CALIBRATION_FRAMES:
                        baseline_ear = np.mean(calibration_ears)
                        EAR_THRESHOLD = baseline_ear * 0.75
                        calibration_done = True
                        print(f"[INFO] EAR calibration complete. Baseline EAR: {baseline_ear:.3f}, Threshold: {EAR_THRESHOLD:.3f}")
                else:
                    # 繪製輪廓
                    cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(mouth)], True, (255, 0, 0), 1)

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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            window_name = "Drowsiness and Yawning Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            target_width = screen_width // 2
            target_height = screen_height // 2
            x_offset = screen_width - target_width
            y_offset = 0

            cv2.resizeWindow(window_name, target_width, target_height)
            cv2.moveWindow(window_name, x_offset, y_offset)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_drowsiness_detection()
