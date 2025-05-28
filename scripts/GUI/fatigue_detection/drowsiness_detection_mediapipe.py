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
    A = euclidean_distance(mouth_landmarks[13], mouth_landmarks[19])  # 上下中心
    B = euclidean_distance(mouth_landmarks[14], mouth_landmarks[18])  # 上下兩側
    C = euclidean_distance(mouth_landmarks[12], mouth_landmarks[16])  # 左右嘴角
    return (A + B) / (2.0 * C)

def start_drowsiness_detection(shared_alert):
    FPS = 30
    TIRED_SECONDS = 2.0
    CONSEC_FRAMES = int(FPS * TIRED_SECONDS)
    COUNTER = 0
    ALARM_ON = False
    ALARM_END_TIME = 0
    ALERT_DURATION = 3

    EAR_CALIBRATION_TIME = 3
    CALIBRATION_FRAMES = int(FPS * EAR_CALIBRATION_TIME)
    calibration_ears = []
    EAR_THRESHOLD = None

    MAR_OPEN_THRESHOLD = 1.2  # 張嘴閾值
    MAR_CLOSE_THRESHOLD = 0.7  # 閉嘴閾值（雙閾值判斷）

    YAWN_COUNT = 0
    yawn_flag = False

    YAWN_ALARM_ON = False  # 哈欠警示旗標
    YAWN_ALERT_COMPLETED = False  # 是否完成三次哈欠警示的標示

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

                mouth_indices = [78, 81, 13, 311, 308, 402, 14, 87, 95, 88, 
                                 178, 317, 82, 81, 80, 191, 88, 178, 87, 14]

                left_eye = [landmarks[i] for i in left_eye_indices]
                right_eye = [landmarks[i] for i in right_eye_indices]
                mouth = [landmarks[i] for i in mouth_indices]

                # 依據 MAR 計算邏輯，選擇對應點
                mouth_mar_landmarks = {
                    12: mouth[5],   # left corner
                    13: mouth[2],   # upper center
                    14: mouth[6],   # upper side
                    16: mouth[4],   # right corner
                    18: mouth[17],  # lower side
                    19: mouth[19]   # lower center
                }

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth_mar_landmarks)

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
                    cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(mouth)], True, (255, 0, 0), 1)

                    # 眼睛疲勞偵測
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
                            # 只有在完成哈欠警示後才重置哈欠計數與狀態
                            if YAWN_ALERT_COMPLETED:
                                YAWN_ALARM_ON = False
                                YAWN_COUNT = 0
                                YAWN_ALERT_COMPLETED = False

                    if ALARM_ON:
                        shared_alert[0] = True
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # 哈欠偵測，雙閾值判斷避免連續計數
                    if mar > MAR_OPEN_THRESHOLD:
                        if not yawn_flag:
                            YAWN_COUNT += 1
                            yawn_flag = True
                    elif mar < MAR_CLOSE_THRESHOLD:
                        yawn_flag = False

                    if YAWN_COUNT >= 3 and not YAWN_ALARM_ON:
                        ALARM_ON = True
                        YAWN_ALARM_ON = True
                        ALARM_END_TIME = time.time() + ALERT_DURATION
                        YAWN_ALERT_COMPLETED = True  # 標示已完成一次三次哈欠警示

                    # 顯示數值資訊
                    cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Yawns: {YAWN_COUNT}", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 顯示畫面與視窗控制
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

