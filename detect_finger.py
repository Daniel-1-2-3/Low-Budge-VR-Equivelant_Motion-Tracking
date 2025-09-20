import cv2
import mediapipe as mp

class Fingers:
    def __init__(self):
        self.mp_hands = mp.solution.hands
        self.hands = mp.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    def identify_trigger_pull(self, frame):
        pass
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w = frame.shape[:2]

    finger_dict = {}

    if results.multi_hand_landmarks:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            lm = hand_lms.landmark

            def up(tip, pip):
                return 1 if lm[tip].y < lm[pip].y else 0

            # Thumb: different rule for left/right
            if label == "Right":
                thumb = 1 if lm[4].x < lm[3].x else 0
            else:
                thumb = 1 if lm[4].x > lm[3].x else 0

            index  = up(8, 6)
            middle = up(12, 10)
            ring   = up(16, 14)
            pinky  = up(20, 18)

            finger_dict[label] = {
                "thumb": thumb,
                "index": index,
                "middle": middle,
                "ring": ring,
                "pinky": pinky
            }

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    print(finger_dict)  # dictionary of finger states each frame

    cv2.imshow("Hands", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
