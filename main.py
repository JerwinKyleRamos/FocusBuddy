import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1) # inverts the camera
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_height, frame_width, _ = frame.shape
    if landmarks_points:
        landmarks = landmarks_points[0].landmark
        for landmark in landmarks[474:478]:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            print(x,y)



    cv2.imshow(' Focus Buddy ', frame)
    cv2.waitKey(1)