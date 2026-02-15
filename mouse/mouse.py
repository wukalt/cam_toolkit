import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np
import math

cam_w, cam_h = 640, 480
smoothening = 7
deadzone = 5
click_delay = 0.3
max_fps = 60

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prev_x, prev_y = 0, 0
pTime = 0

left_pinch_state = "open"
right_pinch_state = "open"
last_click_time = 0

while True:
    start_time = time.time()

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        x_index = int(lm[8].x * cam_w)
        y_index = int(lm[8].y * cam_h)

        x_thumb = int(lm[4].x * cam_w)
        y_thumb = int(lm[4].y * cam_h)

        x_middle = int(lm[12].x * cam_w)
        y_middle = int(lm[12].y * cam_h)

        screen_x = np.interp(x_index, (0, cam_w), (0, screen_w))
        screen_y = np.interp(y_index, (0, cam_h), (0, screen_h))

        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening

        if abs(curr_x - prev_x) > deadzone or abs(curr_y - prev_y) > deadzone:
            pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

        distance_left = math.hypot(x_thumb - x_index, y_thumb - y_index)
        distance_right = math.hypot(x_thumb - x_middle, y_thumb - y_middle)

        pinch_threshold = 35

        if distance_left < pinch_threshold and left_pinch_state == "open":
            if time.time() - last_click_time > click_delay:
                pyautogui.click()
                last_click_time = time.time()
            left_pinch_state = "closed"

        elif distance_left > pinch_threshold:
            left_pinch_state = "open"

        if distance_right < pinch_threshold and right_pinch_state == "open":
            if time.time() - last_click_time > click_delay:
                pyautogui.click(button='right')
                last_click_time = time.time()
            right_pinch_state = "closed"

        elif distance_right > pinch_threshold:
            right_pinch_state = "open"

        cv2.circle(img, (x_index, y_index), 8, (255, 0, 255), -1)
        cv2.circle(img, (x_thumb, y_thumb), 8, (0, 255, 0), -1)
        cv2.circle(img, (x_middle, y_middle), 8, (255, 255, 0), -1)

    elapsed = time.time() - start_time
    delay = max(0, (1 / max_fps) - elapsed)
    time.sleep(delay)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Gesture Mouse Pro", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
