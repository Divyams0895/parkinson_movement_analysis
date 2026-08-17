import cv2
import csv
import time
import os

from src.hand_tracker import HandTracker
from src.features import calculate_hand_features


OUTPUT_FILE = "data/movement_data.csv"


def initialize_csv():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(OUTPUT_FILE):

        with open(
            OUTPUT_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "thumb_index_distance",
                "thumb_middle_distance",
                "thumb_ring_distance",
                "thumb_little_distance",
                "label"
            ])


def main():

    initialize_csv()

    label = input(
        "Enter label (0=control, 1=parkinsonian-pattern): "
    )

    cap = cv2.VideoCapture(0)

    tracker = HandTracker()

    start_time = time.time()

    print("Collecting data...")
    print("Press Q to stop.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = tracker.process(frame)

        tracker.draw_landmarks(
            frame,
            results
        )

        landmarks = tracker.get_landmarks(
            results
        )

        features = calculate_hand_features(
            landmarks
        )

        if features:

            elapsed = time.time() - start_time

            with open(
                OUTPUT_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    elapsed,
                    features["thumb_index_distance"],
                    features["thumb_middle_distance"],
                    features["thumb_ring_distance"],
                    features["thumb_little_distance"],
                    label
                ])

            cv2.putText(
                frame,
                "Recording...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Data Collection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()