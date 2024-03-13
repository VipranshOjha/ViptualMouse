import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

hand_detector = mp.solutions.hands.Hands()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

thumb_index_distance_threshold = 80
thumb_pinky_distance_threshold = 80

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    rgb_frame_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_face = face_mesh.process(rgb_frame_face)
    landmark_points = output_face.multi_face_landmarks

    thumb_x, thumb_y, thumb_tip_x, thumb_tip_y = 0, 0, 0, 0
    index_x, index_y, index_tip_x, index_tip_y = 0, 0, 0, 0
    pinky_x, pinky_y, pinky_tip_x, pinky_tip_y = 0, 0, 0, 0

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    thumb_tip_x = thumb_x
                    thumb_tip_y = thumb_y

                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    index_tip_x = index_x
                    index_tip_y = index_y

                if id == 20:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    pinky_x = screen_width / frame_width * x
                    pinky_y = screen_height / frame_height * y
                    pinky_tip_x = pinky_x
                    pinky_tip_y = pinky_y

    thumb_index_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
    thumb_pinky_distance = ((thumb_x - pinky_x) ** 2 + (thumb_y - pinky_y) ** 2) ** 0.5

    if thumb_index_distance < thumb_index_distance_threshold:
        pyautogui.click(button='left')
        pyautogui.sleep(1)

    if thumb_pinky_distance < thumb_pinky_distance_threshold:
        pyautogui.click(button='right')
        pyautogui.sleep(1)

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_width * landmark.x
                screen_y = screen_height * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow('Vision Pro Lite', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
