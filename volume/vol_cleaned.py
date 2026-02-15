import cv2
import time
import math
import subprocess
from collections import deque
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing


CAM_WIDTH = 520
CAM_HEIGHT = 540
MAX_VOLUME = 80
SMOOTHING_THRESHOLD = 3


def set_volume(percent):
    subprocess.Popen(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def set_mute(state: bool):
    subprocess.Popen(
        ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if state else "0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def map_distance_to_volume(distance):
    mapped = int((distance - 20) / 180 * MAX_VOLUME)
    return max(0, min(mapped, MAX_VOLUME))


def draw_volume_bar(img, volume, muted):
    bar_height = int(volume * 1.7)

    if muted:
        color = (0, 0, 255)
    elif volume < 35:
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(img, (10, 60), (30, 200), (100, 100, 100), 2)
    cv2.rectangle(img, (10, 200), (30, 200 - bar_height), color, cv2.FILLED)
    cv2.putText(img, f"{volume}%", (5, 220),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)


def draw_status(img, muted, volume, width):
    if muted:
        text, color = "MUTE", (0, 0, 255)
    elif volume < 40:
        text, color = "MED", (0, 255, 255)
    else:
        text, color = "HIGH", (0, 255, 0)

    cv2.putText(img, text, (width - 80, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.8, color, 2)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    hands_detector = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    trail = deque(maxlen=10)

    current_volume = 40
    muted = False
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark

            x4, y4 = int(lm[4].x * w), int(lm[4].y * h)
            x8, y8 = int(lm[8].x * w), int(lm[8].y * h)

            distance = math.hypot(x8 - x4, y8 - y4)

            trail.append((x8, y8))
            for i, (tx, ty) in enumerate(trail):
                size = int(3 + (i / len(trail)) * 10)
                cv2.circle(frame, (tx, ty), size, (200, 100, 255), -1)

            if y8 > y4 + 40:
                if not muted:
                    muted = True
                    set_mute(True)
            else:
                if muted:
                    muted = False
                    set_mute(False)

            if not muted:
                new_volume = map_distance_to_volume(distance)
                if abs(new_volume - current_volume) > SMOOTHING_THRESHOLD:
                    current_volume = new_volume
                    set_volume(current_volume)

            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

        draw_volume_bar(frame, current_volume, muted)
        draw_status(frame, muted, current_volume, w)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 255), 2)

        cv2.imshow("Gesture Volume Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
