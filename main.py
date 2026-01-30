from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame
from collections import deque

pygame.mixer.init()
pygame.mixer.music.load('Alarm/mixkit-facility-alarm-sound-999.wav')

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) #outputs all 478 face landmarks

HISTORY_LENGTH = 5
h_ratio_history = deque(maxlen=HISTORY_LENGTH)
v_ratio_history = deque(maxlen=HISTORY_LENGTH)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1) # inverts the camera
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_height, frame_width, _ = frame.shape

    is_looking_at_screen = True  # default assumption

    if landmarks_points:
        landmarks = landmarks_points[0].landmark

        #----------Right Eye Landmark-------------
        right_pupil_points = []
        for landmark in landmarks[473:478]: #right eye
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            right_pupil_points.append((x, y))

        right_eye_corner_points = []
        for i in (362, 263, 386, 374):
            x = int(landmarks[i].x * frame_width)
            y = int(landmarks[i].y * frame_height)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
            right_eye_corner_points.append((x, y))

        right_inner = right_eye_corner_points[1]
        right_outer = right_eye_corner_points[0]
        right_upper = right_eye_corner_points[2]
        right_lower = right_eye_corner_points[3]

        # ----------Left Eye Landmark-------------
        left_pupil_points = []
        for landmark in landmarks[468:473]: #left eye
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
            left_pupil_points.append((x, y))

        left_eye_corner_points = []
        for i in (33, 133, 159, 145):
            x = int(landmarks[i].x * frame_width)
            y = int(landmarks[i].y * frame_height)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
            left_eye_corner_points.append((x, y))

        left_inner = left_eye_corner_points[1]
        left_outer = left_eye_corner_points[0]
        left_upper = left_eye_corner_points[2]
        left_lower = left_eye_corner_points[3]

        # ----------------Calculates center of the Iris----------
        left_pupil_center = np.mean(left_pupil_points, axis=0).astype(int)
        right_pupil_center = np.mean(right_pupil_points, axis=0).astype(int)

        # --------------- HEAD POSITION LANDMARK ----------
        # (LEFT AND RIGHT)
        left_face_x = int(landmarks[234].x * frame_width)
        right_face_x = int(landmarks[454].x * frame_width)
        face_center_x = (left_face_x + right_face_x) / 2

        # (UP AND DOWN)
        left_face_y = int(landmarks[10].y * frame_height)
        right_face_y = int(landmarks[152].y * frame_height)
        face_center_y = (left_face_y + right_face_y) / 2

        #------------------ NOSE ----------------
        nose_x = int(landmarks[1].x * frame_width)
        nose_y = int(landmarks[1].y * frame_height)

        #-------------- FACE DIMENSIONS -------------
        face_width_x = abs(right_face_x - left_face_x)
        face_height_y = abs(right_face_y - left_face_y)

        # ---------------- HORIZONTAL TURN -------------------
        if face_width_x != 0:
            turn_ratio_x = abs(nose_x - face_center_x)/ face_width_x
        else:
            turn_ratio_x = 0

        # ---------------- VERTICAL TURN -------------------
        if face_height_y != 0:
            turn_ratio_y = abs(nose_y - face_center_y) / face_height_y
        else:
            turn_ratio_y = 0

        head_facing_screen = (turn_ratio_x < 0.80) and (turn_ratio_y < 0.80)

        # Detects what direction your pupils are facing
        def gaze_ratio_horizontal(iris, left_corner, right_corner):
            eye_width = np.linalg.norm(np.array(right_corner) - np.array(left_corner))
            if eye_width == 0:
                return 0.5
            return np.linalg.norm(np.array(iris) - np.array(left_corner)) / eye_width

        def gaze_ratio_vertical(iris, top, bottom):
            eye_height = np.linalg.norm(np.array(bottom) - np.array(top))
            if eye_height == 0:
                return 0.5
            return np.linalg.norm(np.array(iris) - np.array(top)) / eye_height

        # Horizontal iris ratios
        left_h = gaze_ratio_horizontal(left_pupil_center, left_outer, left_inner)
        right_h = gaze_ratio_horizontal(right_pupil_center, right_inner, right_outer)
        avg_h = (left_h + right_h) / 2

        # Vertical iris ratios
        left_v = gaze_ratio_vertical(left_pupil_center, left_upper,left_lower)
        right_v = gaze_ratio_vertical(right_pupil_center, right_upper, right_lower)
        avg_v = (left_v + right_v) / 2
        # print(avg_v)

        h_ratio_history.append(avg_h)
        v_ratio_history.append(avg_v)
        smooth_h = np.mean(h_ratio_history)
        smooth_v = np.mean(v_ratio_history)

        print(smooth_h, smooth_v)

        # Eyes centered thresholds
        eyes_centered = (0.20 < smooth_h < 0.70) and (0.49 < smooth_v < 0.65)

        # --- FINAL DECISION ---
        if eyes_centered:
            is_looking_at_screen = True
            if turn_ratio_x > head_facing_screen or turn_ratio_y > head_facing_screen:
                is_looking_at_screen = False
        else:
            is_looking_at_screen = False


        if is_looking_at_screen:
            status_text = "LOOKING AT SCREEN"
        else:
            status_text = "LOOKING AWAY"
        cv2.putText(frame, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(' Focus Buddy ', frame)
    cv2.waitKey(1)