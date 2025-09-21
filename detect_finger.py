import cv2
import mediapipe as mp
import math

class Fingers:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    def detect_left_fist(self, frame):
        """
        Detects if the LEFT hand is making a fist.
        Returns True if left hand fist is detected, False otherwise.
        """
        def distance(p1, p2):
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                # Only check LEFT hand for fist
                if label == "Left":
                    lm = hand_lms.landmark
                    
                    # Calculate distances from fingertips to palm center (landmark 9)
                    palm_center = lm[9]
                    
                    # Distance from each fingertip to palm
                    thumb_to_palm = distance(lm[4], palm_center)
                    index_to_palm = distance(lm[8], palm_center)
                    middle_to_palm = distance(lm[12], palm_center)
                    ring_to_palm = distance(lm[16], palm_center)
                    pinky_to_palm = distance(lm[20], palm_center)
                    
                    # Calculate finger curl ratios
                    thumb_curl = thumb_to_palm / distance(lm[3], palm_center)
                    index_curl = index_to_palm / distance(lm[6], palm_center)  
                    middle_curl = middle_to_palm / distance(lm[10], palm_center)
                    ring_curl = ring_to_palm / distance(lm[14], palm_center)
                    pinky_curl = pinky_to_palm / distance(lm[18], palm_center)
                    
                    curl_threshold = 1.3
                    
                    fingers_curled = [
                        thumb_curl < curl_threshold,
                        index_curl < curl_threshold,
                        middle_curl < curl_threshold, 
                        ring_curl < curl_threshold,
                        pinky_curl < curl_threshold
                    ]
                    
                    # Check if fingertips are close together
                    fingertip_distances = [
                        distance(lm[4], lm[8]),   # thumb to index
                        distance(lm[8], lm[12]),  # index to middle  
                        distance(lm[12], lm[16]), # middle to ring
                        distance(lm[16], lm[20])  # ring to pinky
                    ]
                    
                    avg_fingertip_distance = sum(fingertip_distances) / len(fingertip_distances)
                    compact_threshold = 0.08
                    
                    if sum(fingers_curled) >= 4 and avg_fingertip_distance < compact_threshold:
                        return True
        
        return False
    
    def detect_right_index_trigger(self, frame):
        """
        Improved trigger pull detection using multiple methods.
        Works better when index finger is facing camera in gun grip position.
        """
        def distance(p1, p2):
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        def calculate_angle(p1, p2, p3):
            """Calculate angle at p2 formed by points p1, p2, p3"""
            v1 = (p1.x - p2.x, p1.y - p2.y)
            v2 = (p3.x - p2.x, p3.y - p2.y)
            
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 * mag2 == 0:
                return 180
            
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
            angle = math.degrees(math.acos(cos_angle))
            return angle
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                # Only check RIGHT hand for trigger pull
                if label == "Right":
                    lm = hand_lms.landmark
                    
                    # METHOD 1: Joint angle analysis
                    # Calculate angles at PIP (6) and DIP (7) joints
                    pip_angle = calculate_angle(lm[5], lm[6], lm[7])  # MCP-PIP-DIP angle
                    dip_angle = calculate_angle(lm[6], lm[7], lm[8])  # PIP-DIP-TIP angle
                    
                    # METHOD 2: Z-coordinate analysis (depth)
                    # When finger curls, tip moves closer to camera (z decreases)
                    z_diff = lm[6].z - lm[8].z  # PIP z - TIP z
                    
                    # METHOD 3: Distance ratio analysis
                    # Compare tip-to-wrist vs pip-to-wrist distance
                    wrist = lm[0]
                    tip_to_wrist = distance(lm[8], wrist)
                    pip_to_wrist = distance(lm[6], wrist)
                    distance_ratio = tip_to_wrist / pip_to_wrist if pip_to_wrist > 0 else 1
                    
                    # METHOD 4: Tip proximity to middle finger base
                    # When trigger pulled, index tip gets closer to middle finger base
                    tip_to_middle_base = distance(lm[8], lm[9])
                    
                    # METHOD 5: Check if other fingers are extended (gun grip)
                    # In gun grip, middle, ring, pinky should be more extended
                    middle_curl = distance(lm[12], lm[0]) / distance(lm[10], lm[0])
                    ring_curl = distance(lm[16], lm[0]) / distance(lm[14], lm[0])
                    pinky_curl = distance(lm[20], lm[0]) / distance(lm[18], lm[0])
                    
                    other_fingers_extended = (middle_curl > 0.9 and ring_curl > 0.9) or \
                                           (middle_curl > 0.9 and pinky_curl > 0.9)
                    
                    # Trigger detection logic - more sensitive thresholds
                    trigger_pulled = False
                    
                    # Primary detection: Joint angles
                    if pip_angle < 160 or dip_angle < 160:  # Bent joints
                        trigger_pulled = True
                    
                    # Secondary detection: Distance ratio
                    elif distance_ratio < 0.85:  # Tip closer to wrist than normal
                        trigger_pulled = True
                    
                    # Tertiary detection: Z-coordinate (if available and reliable)
                    elif z_diff > 0.02:  # Tip is notably closer to camera
                        trigger_pulled = True
                    
                    # Quaternary detection: Tip proximity in gun grip
                    elif tip_to_middle_base < 0.15 and other_fingers_extended:
                        trigger_pulled = True
                    
                    # Debug info (optional)
                    if False:  # Set to True for debugging
                        print(f"PIP angle: {pip_angle:.1f}, DIP angle: {dip_angle:.1f}")
                        print(f"Distance ratio: {distance_ratio:.3f}")
                        print(f"Z diff: {z_diff:.3f}")
                        print(f"Tip to middle base: {tip_to_middle_base:.3f}")
                    
                    return trigger_pulled
        
        return False
    
    def identify_trigger_pull(self, frame):
        """
        Main function that combines left fist detection and right trigger detection.
        Returns a tuple: (left_fist_detected, right_trigger_pulled)
        """
        left_fist = self.detect_left_fist(frame)
        right_trigger = self.detect_right_index_trigger(frame)
        return left_fist, right_trigger
    
if __name__ == "__main__":
    finger_signs = Fingers()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # For smoothing detection (reduce jitter)
    trigger_history = []
    history_size = 3
    
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check for left fist and right trigger pull
        left_fist, right_trigger = finger_signs.identify_trigger_pull(frame)
        
        # Smooth trigger detection using history
        trigger_history.append(right_trigger)
        if len(trigger_history) > history_size:
            trigger_history.pop(0)
        
        # Require majority of recent frames to show trigger
        smoothed_trigger = sum(trigger_history) > history_size // 2
        
        results = finger_signs.hands.process(rgb)
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
                    
                    # Draw index finger joints for visualization
                    # This helps see the trigger finger movement
                    for idx in [5, 6, 7, 8]:  # Index finger joints
                        x = int(lm[idx].x * w)
                        y = int(lm[idx].y * h)
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                        if idx > 5:  # Draw lines between joints
                            x_prev = int(lm[idx-1].x * w)
                            y_prev = int(lm[idx-1].y * h)
                            cv2.line(frame, (x_prev, y_prev), (x, y), (255, 0, 0), 2)
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
                    frame, hand_lms, finger_signs.mp_hands.HAND_CONNECTIONS)

        # Display detection status
        status_text = []
        if left_fist:
            status_text.append("LEFT FIST DETECTED!")
        if smoothed_trigger:
            status_text.append("RIGHT TRIGGER PULLED!")
        
        if not status_text:
            status_text = ["No gestures detected"]
            
        # Display status text
        for i, text in enumerate(status_text):
            color = (0, 255, 0) if text != "No gestures detected" else (0, 0, 255)
            cv2.putText(frame, text, (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show instructions
        cv2.putText(frame, "Hold gun grip with right hand, curl index to trigger", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' or ESC to quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        print(f"Finger states: {finger_dict}, Left fist: {left_fist}, Right trigger: {smoothed_trigger}")

        cv2.imshow("Hands", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()