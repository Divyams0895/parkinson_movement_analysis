import numpy as np


def euclidean_distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_hand_features(landmarks):

    if landmarks is None:
        return None

    # MediaPipe landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    little_tip = landmarks[20]

    # Thumb-index distance
    thumb_index_distance = euclidean_distance(
        thumb_tip,
        index_tip
    )

    # Other finger distances
    thumb_middle_distance = euclidean_distance(
        thumb_tip,
        middle_tip
    )

    thumb_ring_distance = euclidean_distance(
        thumb_tip,
        ring_tip
    )

    thumb_little_distance = euclidean_distance(
        thumb_tip,
        little_tip
    )

    return {
        "thumb_index_distance": thumb_index_distance,
        "thumb_middle_distance": thumb_middle_distance,
        "thumb_ring_distance": thumb_ring_distance,
        "thumb_little_distance": thumb_little_distance
    }