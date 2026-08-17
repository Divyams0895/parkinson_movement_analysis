import cv2
import mediapipe as mp


class HandTracker:

    def __init__(
        self,
        max_num_hands=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        return results

    def draw_landmarks(self, frame, results):

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame

    def get_landmarks(self, results):

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for landmark in hand.landmark:

            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return landmarks