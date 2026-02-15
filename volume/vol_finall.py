import cv2
import mediapipe as mp
import time
import math
import subprocess
from collections import deque


cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

pTime = 0
MAX_VOLUME = 80
isMuted = False
current_volume = 40

trail = deque(maxlen=10)

def set_volume(percent):
    subprocess.Popen(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])

def set_mute(on):
    subprocess.Popen(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if on else "0"])

def draw_status_icon(img, is_muted, volume_level, width):
    if is_muted:
        text, color = "MUTE", (0, 0, 255)
    elif volume_level == 0:
        text, color = "LOW", (100, 100, 255)
    elif volume_level < 40:
        text, color = "MED", (0, 255, 255)
    else:
        text, color = "HIGH", (0, 255, 0)
    
    cv2.putText(img, text, (width - 78, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, color, 2)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        x4, y4 = int(lm[4].x * w), int(lm[4].y * h)
        x8, y8 = int(lm[8].x * w), int(lm[8].y * h)

        distance = math.hypot(x8 - x4, y8 - y4)

        trail.append((x8, y8))
        for i, (tx, ty) in enumerate(trail):
            size = int(3 + (i / len(trail)) * 10)     
            alpha = i / len(trail)
            color = (int(180 * alpha), int(100 * alpha), 255)  
            cv2.circle(img, (tx, ty), size, color, -1)

        if y8 > y4 + 40:
            if not isMuted:
                set_mute(True)
                isMuted = True
            cv2.putText(img, "MUTED", (10, 220), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            if isMuted:
                set_mute(False)
                isMuted = False

        if not isMuted:
            mapped = int((distance - 20) / 180 * MAX_VOLUME)
            mapped = max(0, min(mapped, MAX_VOLUME))
            if abs(mapped - current_volume) > 3:
                current_volume = mapped
                set_volume(mapped)

        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS,
                              mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2),
                              mpDraw.DrawingSpec(color=(200, 100, 255), thickness=2))

    bar_height = int(current_volume * 1.7)
    if isMuted:
        hud_color = (0, 0, 255)
    elif current_volume == 0:
        hud_color = (80, 80, 80)
    elif current_volume < 35:
        hud_color = (0, 255, 255)
    else:
        hud_color = (0, 255, 0)

    cv2.rectangle(img, (10, 200), (30, 200 - bar_height), hud_color, cv2.FILLED)
    cv2.rectangle(img, (10, 60), (30, 200), (100, 100, 100), 2)
    cv2.putText(img, f"{current_volume}%", (5, 215), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

    draw_status_icon(img, isMuted, current_volume, w)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    cv2.imshow("Hand Volume + Trail", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
