import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

keys = [
    ["1","2","3","4","5","6","7","8","9","0","-","="],
    ["Q","W","E","R","T","Y","U","I","O","P","[","]"],
    ["A","S","D","F","G","H","J","K","L",";","'"],
    ["Z","X","C","V","B","N","M",",",".","/"],
    ["SPACE","BACK","ENTER","SHIFT","CAPS"]
]

key_width = 55
key_height = 55
start_x = 40
start_y = 120

typed_text = ""
caps = False
shift = False

finger_ids = [4,8,12,16,20]
finger_history = {}
finger_state = {}

cap.set(3, 520)
cap.set(4, 340)


for fid in finger_ids:
    finger_history[fid] = None
    finger_state[fid] = False

last_press_time = 0
delay = 0.25


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    current_fingers = {}

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id in finger_ids:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    current_fingers[id] = (cx, cy)
                    cv2.circle(img, (cx, cy), 8, (0,255,0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    key_positions = []

    for i, row in enumerate(keys):
        for j, key in enumerate(row):

            width = key_width
            if key == "SPACE":
                width = key_width * 5
            if key in ["ENTER"]:
                width = key_width * 2

            x = start_x + j * key_width
            y = start_y + i * key_height

            cv2.rectangle(img, (x,y), (x+width,y+key_height), (255,0,255), 2)
            cv2.putText(img, key, (x+5,y+35),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

            key_positions.append((key,x,y,x+width,y+key_height))

    current_time = time.time()

    for fid, pos in current_fingers.items():

        if finger_history[fid] is not None:
            prev_x, prev_y = finger_history[fid]
            curr_x, curr_y = pos

            dy = curr_y - prev_y

            if dy > 15 and not finger_state[fid]:

                for key, x1,y1,x2,y2 in key_positions:
                    if x1 < curr_x < x2 and y1 < curr_y < y2:

                        if current_time - last_press_time > delay:

                            if key == "SPACE":
                                typed_text += " "

                            elif key == "BACK":
                                typed_text = typed_text[:-1]

                            elif key == "ENTER":
                                typed_text += "\n"

                            elif key == "SHIFT":
                                shift = True

                            elif key == "CAPS":
                                caps = not caps

                            else:
                                if caps or shift:
                                    typed_text += key.upper()
                                else:
                                    typed_text += key.lower()
                                shift = False

                            last_press_time = current_time
                            finger_state[fid] = True
                        break

            if dy < -10:
                finger_state[fid] = False

        finger_history[fid] = pos

    cv2.rectangle(img, (40, 40), (1000, 100), (0,0,0), cv2.FILLED)
    cv2.putText(img, typed_text[-40:], (50, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv2.imshow("Professional Virtual Keyboard", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
