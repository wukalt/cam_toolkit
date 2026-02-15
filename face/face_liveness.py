import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import numpy as np

# تنظیمات
BLINK_THRESHOLD = 0.20   # آستانهٔ نسبت چشم (EAR)
CLOSED_FRAMES = 2        # چند فریم پشت‌سر‌هم چشم بسته باشد
REQUIRED_BLINKS = 1      # چند چشمک برای تأیید لایونس
WINDOW_SECONDS = 5       # بازهٔ زمانی مجاز برای چشمک‌ها

detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(0)

blink_count = 0
consec_closed = 0
blink_timestamps = []

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]

        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]

        left_eye = np.array([face[i] for i in left_eye_idx])
        right_eye = np.array([face[i] for i in right_eye_idx])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < BLINK_THRESHOLD:
            consec_closed += 1
        else:
            if consec_closed >= CLOSED_FRAMES:
                blink_count += 1
                blink_timestamps.append(time.time())
                blink_timestamps = [t for t in blink_timestamps if (time.time() - t) <= WINDOW_SECONDS]
                print(f"Blink #{blink_count} detected.")
            consec_closed = 0

        liveness_ok = len(blink_timestamps) >= REQUIRED_BLINKS
        status = "LIVENESS: ✅ OK" if liveness_ok else "LIVENESS: ⏳ WAITING"

        cv2.putText(img, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(img, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0) if liveness_ok else (0,165,255), 2)

    cv2.imshow("Liveness via Blink - cvzone", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
