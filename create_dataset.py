import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

HEALTHY_DIR = "data/healthy"
PARKINSON_DIR = "data/parkinson"

OUTPUT_FILE = "data/features.csv"


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# DISTANCE
# ============================================================

def calculate_distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# EXTRACT VIDEO FEATURES
# ============================================================

def extract_video_features(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    distances = []

    frame_number = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                continue

            hand = results.multi_hand_landmarks[0]

            thumb = hand.landmark[
                mp_hands.HandLandmark.THUMB_TIP
            ]

            index = hand.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            middle = hand.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]

            ring = hand.landmark[
                mp_hands.HandLandmark.RING_FINGER_TIP
            ]

            little = hand.landmark[
                mp_hands.HandLandmark.PINKY_TIP
            ]

            thumb_point = [
                thumb.x,
                thumb.y
            ]

            index_point = [
                index.x,
                index.y
            ]

            middle_point = [
                middle.x,
                middle.y
            ]

            ring_point = [
                ring.x,
                ring.y
            ]

            little_point = [
                little.x,
                little.y
            ]

            distances.append({

                "thumb_index":
                    calculate_distance(
                        thumb_point,
                        index_point
                    ),

                "thumb_middle":
                    calculate_distance(
                        thumb_point,
                        middle_point
                    ),

                "thumb_ring":
                    calculate_distance(
                        thumb_point,
                        ring_point
                    ),

                "thumb_little":
                    calculate_distance(
                        thumb_point,
                        little_point
                    )
            })

    cap.release()

    if len(distances) < 10:

        print(
            "Not enough hand detections:",
            video_path
        )

        return None

    return pd.DataFrame(distances), fps


# ============================================================
# CALCULATE FEATURES
# ============================================================

def calculate_features(data, fps):

    distance = data[
        "thumb_index"
    ].values

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_distance = np.mean(
        distance
    )

    std_distance = np.std(
        distance
    )

    min_distance = np.min(
        distance
    )

    max_distance = np.max(
        distance
    )

    movement_range = (
        max_distance -
        min_distance
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    if len(distance) > 1:

        velocity = (
            np.diff(distance) * fps
        )

        mean_velocity = np.mean(
            np.abs(velocity)
        )

        std_velocity = np.std(
            velocity
        )

    else:

        mean_velocity = 0.0
        std_velocity = 0.0

    # --------------------------------------------------------
    # Tapping
    # --------------------------------------------------------

    threshold = 0.05

    closed = distance < threshold

    tap_count = 0

    previous_state = False

    for state in closed:

        if state and not previous_state:

            tap_count += 1

        previous_state = state

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = len(distance) / fps

    if duration > 0:

        taps_per_second = (
            tap_count / duration
        )

    else:

        taps_per_second = 0.0

    # --------------------------------------------------------
    # Movement variability
    # --------------------------------------------------------

    if len(distance) > 1:

        movement_variability = np.std(
            np.diff(distance)
        )

    else:

        movement_variability = 0.0

    # --------------------------------------------------------
    # Return feature vector
    # --------------------------------------------------------

    return {

        "mean_distance":
            mean_distance,

        "std_distance":
            std_distance,

        "movement_range":
            movement_range,

        "mean_velocity":
            mean_velocity,

        "std_velocity":
            std_velocity,

        "tap_count":
            tap_count,

        "taps_per_second":
            taps_per_second,

        "movement_variability":
            movement_variability
    }


# ============================================================
# PROCESS FOLDER
# ============================================================

def process_folder(
    folder,
    label,
    label_name
):

    rows = []

    if not os.path.exists(folder):

        print(
            f"\nFolder does not exist: {folder}"
        )

        return rows

    video_files = [

        file for file in os.listdir(folder)

        if file.lower().endswith(
            (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv"
            )
        )
    ]

    print(
        f"\nFound {len(video_files)} "
        f"{label_name} videos."
    )

    for index, filename in enumerate(
        video_files,
        start=1
    ):

        path = os.path.join(
            folder,
            filename
        )

        print(
            f"[{index}/{len(video_files)}] "
            f"Processing {filename}..."
        )

        result = extract_video_features(path)

        if result is None:
            print("  Skipped.")
            continue

        data, fps = result

        features = calculate_features(
            data,
            fps
        )

        features["label"] = label

        features["video"] = filename

        rows.append(
            features
        )

        print(
            "  Done."
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "CREATING PARKINSON MOVEMENT DATASET"
    )
    print("=" * 60)

    healthy_rows = process_folder(
        HEALTHY_DIR,
        0,
        "Healthy"
    )

    parkinson_rows = process_folder(
        PARKINSON_DIR,
        1,
        "Parkinson"
    )

    rows = (
        healthy_rows +
        parkinson_rows
    )

    if len(rows) == 0:

        print(
            "\nNo usable videos were found."
        )

        print(
            "\nExpected:"
        )

        print(
            "data/videos/healthy/"
        )

        print(
            "data/videos/parkinson/"
        )

        return

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    columns = [

        "mean_distance",
        "std_distance",
        "movement_range",
        "mean_velocity",
        "std_velocity",
        "tap_count",
        "taps_per_second",
        "movement_variability",
        "label",
        "video"
    ]

    df = df[
        columns
    ]

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)

    print(
        "DATASET CREATED"
    )

    print("=" * 60)

    print(
        f"\nTotal samples: {len(df)}"
    )

    print(
        f"Healthy samples: "
        f"{sum(df['label'] == 0)}"
    )

    print(
        f"Parkinson samples: "
        f"{sum(df['label'] == 1)}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        df.head()
    )


if __name__ == "__main__":

    main()