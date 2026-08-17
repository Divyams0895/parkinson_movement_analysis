import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Early Parkinson Detection",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "models/parkinson_screening_model.pkl"

# Label convention used by the model:
# 0 = Healthy
# 1 = Parkinsonian pattern


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def extract_video_features(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frame_number = 0

    records = []

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

            # ------------------------------------------------
            # Convert BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # MediaPipe processing
            # ------------------------------------------------

            results = hands.process(
                rgb_frame
            )

            if not results.multi_hand_landmarks:
                continue

            hand = results.multi_hand_landmarks[0]

            # ------------------------------------------------
            # Get landmarks
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Coordinates
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Calculate distances
            # ------------------------------------------------

            thumb_index_distance = calculate_distance(
                thumb_point,
                index_point
            )

            thumb_middle_distance = calculate_distance(
                thumb_point,
                middle_point
            )

            thumb_ring_distance = calculate_distance(
                thumb_point,
                ring_point
            )

            thumb_little_distance = calculate_distance(
                thumb_point,
                little_point
            )

            records.append({

                "frame": frame_number,

                "time": frame_number / fps,

                "thumb_x": thumb.x,
                "thumb_y": thumb.y,

                "index_x": index.x,
                "index_y": index.y,

                "middle_x": middle.x,
                "middle_y": middle.y,

                "ring_x": ring.x,
                "ring_y": ring.y,

                "little_x": little.x,
                "little_y": little.y,

                "thumb_index_distance":
                    thumb_index_distance,

                "thumb_middle_distance":
                    thumb_middle_distance,

                "thumb_ring_distance":
                    thumb_ring_distance,

                "thumb_little_distance":
                    thumb_little_distance
            })

    cap.release()

    if len(records) < 10:

        return None, None

    data = pd.DataFrame(records)

    return data, fps


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_features(data, fps):

    distance = data[
        "thumb_index_distance"
    ].values

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_distance = np.mean(distance)

    std_distance = np.std(distance)

    min_distance = np.min(distance)

    max_distance = np.max(distance)

    movement_range = (
        max_distance -
        min_distance
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    if len(distance) > 1:

        velocity = np.diff(distance) * fps

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
    # Finger tapping
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

    duration = (
        data["time"].iloc[-1]
        -
        data["time"].iloc[0]
    )

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
    # Additional aggregate features
    # --------------------------------------------------------

    features = {

        # New model features
        "mean_distance":
            mean_distance,

        "std_distance":
            std_distance,

        "min_distance":
            min_distance,

        "max_distance":
            max_distance,

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
            movement_variability,

        # Legacy model features
        # These allow compatibility with your
        # previous 4-feature model.

        "thumb_index_distance":
            np.mean(
                data["thumb_index_distance"]
            ),

        "thumb_middle_distance":
            np.mean(
                data["thumb_middle_distance"]
            ),

        "thumb_ring_distance":
            np.mean(
                data["thumb_ring_distance"]
            ),

        "thumb_little_distance":
            np.mean(
                data["thumb_little_distance"]
            )
    }

    return features


# ============================================================
# MODEL PREDICTION
# ============================================================

def make_prediction(features):

    if not os.path.exists(MODEL_PATH):

        return {
            "status": "no_model"
        }

    try:

        model = joblib.load(
            MODEL_PATH
        )

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

    # --------------------------------------------------------
    # Determine the features expected by the model
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        model_features = list(
            model.feature_names_in_
        )

    else:

        # Fallback for models saved without
        # feature names.

        model_features = [
            "thumb_index_distance",
            "thumb_middle_distance",
            "thumb_ring_distance",
            "thumb_little_distance"
        ]

    # --------------------------------------------------------
    # Check whether all required features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in features
    ]

    if missing_features:

        return {
            "status": "feature_error",
            "missing": missing_features,
            "expected": model_features
        }

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[
            features[feature]
            for feature in model_features
        ]],
        columns=model_features
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(X)[0]

    except Exception as e:

        return {
            "status": "prediction_error",
            "message": str(e)
        }

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    confidence = None

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(X)[0]

            confidence = float(
                np.max(probabilities)
            )

        except Exception:

            confidence = None

    # --------------------------------------------------------
    # Convert prediction to integer if possible
    # --------------------------------------------------------

    try:

        numeric_prediction = int(
            prediction
        )

    except Exception:

        numeric_prediction = prediction

    # --------------------------------------------------------
    # Interpret result
    # --------------------------------------------------------

    if (
        numeric_prediction == 1
        or str(prediction).lower()
        in [
            "parkinson",
            "parkinsons",
            "parkinsonian",
            "patient",
            "1"
        ]
    ):

        result = "PARKINSON"

        result_description = (
            "Parkinsonian movement pattern detected"
        )

    else:

        result = "HEALTHY"

        result_description = (
            "Healthy movement pattern detected"
        )

    return {

        "status": "success",

        "prediction":
            numeric_prediction,

        "result":
            result,

        "description":
            result_description,

        "confidence":
            confidence,

        "model_features":
            model_features,

        "probabilities":
            probabilities
    }


# ============================================================
# CUSTOM RESULT DISPLAY
# ============================================================

def display_result(result):

    st.subheader(
        "🧠 Screening Result"
    )

    if result["result"] == "PARKINSON":

        st.error(
            "🔴 PARKINSON"
        )

        st.markdown(
            """
            ### Parkinsonian movement pattern detected

            The model classified the uploaded movement
            as belonging to the Parkinsonian-pattern class.
            """
        )

    else:

        st.success(
            "🟢 HEALTHY"
        )

        st.markdown(
            """
            ### Healthy movement pattern detected

            The model classified the uploaded movement
            as belonging to the healthy/control class.
            """
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if result["confidence"] is not None:

        confidence_percent = (
            result["confidence"] * 100
        )

        st.metric(
            "Model Confidence",
            f"{confidence_percent:.1f}%"
        )

        st.progress(
            result["confidence"]
        )

    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    st.warning(
        """
        ⚠️ This result is generated by a machine-learning
        research prototype. It is NOT a medical diagnosis.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 Early Parkinson Detection"
)

st.markdown(
    """
    ### MediaPipe-Based Movement Screening

    Upload a person's hand/finger movement video.
    The system extracts movement features using
    MediaPipe and applies a trained machine-learning
    model.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Test Settings"
)

st.sidebar.markdown(
    """
    ### Recommended Video

    • One hand clearly visible

    • Good lighting

    • Camera facing the hand

    • Finger tapping or controlled hand movement

    • Approximately 20–30 seconds

    ### Recommended Position

    Keep the hand inside the camera frame
    throughout the test.
    """
)

st.sidebar.markdown(
    "---"
)

if os.path.exists(MODEL_PATH):

    st.sidebar.success(
        "✅ ML model found"
    )

else:

    st.sidebar.error(
        "❌ ML model not found"
    )

    st.sidebar.caption(
        f"Expected location: {MODEL_PATH}"
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Movement Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ============================================================
# PROCESS UPLOADED VIDEO
# ============================================================

if uploaded_file is not None:

    st.subheader(
        "🎥 Uploaded Video"
    )

    st.video(
        uploaded_file
    )

    # --------------------------------------------------------
    # Create temporary file
    # --------------------------------------------------------

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    if not file_extension:

        file_extension = ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        video_path = temp_file.name

    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Movement",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(
            0
        )

        status = st.empty()

        # ----------------------------------------------------
        # Step 1
        # ----------------------------------------------------

        status.info(
            "Step 1/4 — Reading video and detecting hand landmarks..."
        )

        progress.progress(
            25
        )

        data, fps = extract_video_features(
            video_path
        )

        # ----------------------------------------------------
        # Check detection
        # ----------------------------------------------------

        if data is None:

            progress.progress(
                100
            )

            status.error(
                "Analysis failed."
            )

            st.error(
                """
                ❌ Could not detect enough hand landmarks.

                Please upload another video where:

                • The hand is clearly visible
                • Lighting is sufficient
                • The camera is not too far away
                • Fingers are not heavily occluded
                • The video contains actual hand movement
                """
            )

        else:

            # ------------------------------------------------
            # Step 2
            # ------------------------------------------------

            status.info(
                "Step 2/4 — Calculating movement features..."
            )

            progress.progress(
                50
            )

            features = calculate_features(
                data,
                fps
            )

            # ------------------------------------------------
            # Movement Analysis
            # ------------------------------------------------

            st.subheader(
                "📊 Movement Analysis"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Detected Frames",
                f"{len(data):,}"
            )

            col2.metric(
                "Tap Count",
                features["tap_count"]
            )

            col3.metric(
                "Taps / Second",
                f'{features["taps_per_second"]:.2f}'
            )

            col4.metric(
                "Movement Variability",
                f'{features["movement_variability"]:.4f}'
            )

            # ------------------------------------------------
            # Additional metrics
            # ------------------------------------------------

            col5, col6, col7, col8 = st.columns(4)

            col5.metric(
                "Mean Distance",
                f'{features["mean_distance"]:.4f}'
            )

            col6.metric(
                "Movement Range",
                f'{features["movement_range"]:.4f}'
            )

            col7.metric(
                "Mean Velocity",
                f'{features["mean_velocity"]:.4f}'
            )

            col8.metric(
                "Video FPS",
                f"{fps:.1f}"
            )

            # ------------------------------------------------
            # Step 3
            # ------------------------------------------------

            status.info(
                "Step 3/4 — Generating movement graph..."
            )

            progress.progress(
                70
            )

            # ------------------------------------------------
            # Movement graph
            # ------------------------------------------------

            st.subheader(
                "📈 Thumb–Index Movement"
            )

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                data["time"],
                data["thumb_index_distance"],
                linewidth=1.5
            )

            ax.set_xlabel(
                "Time (seconds)"
            )

            ax.set_ylabel(
                "Thumb–Index Distance"
            )

            ax.set_title(
                "Hand Movement Signal"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            st.pyplot(
                fig,
                clear_figure=True
            )

            # ------------------------------------------------
            # Tap visualization
            # ------------------------------------------------

            st.subheader(
                "👆 Finger Tapping Signal"
            )

            fig2, ax2 = plt.subplots(
                figsize=(12, 3)
            )

            ax2.plot(
                data["time"],
                data["thumb_index_distance"]
            )

            ax2.axhline(
                y=0.05,
                linestyle="--",
                label="Tap threshold"
            )

            ax2.set_xlabel(
                "Time (seconds)"
            )

            ax2.set_ylabel(
                "Distance"
            )

            ax2.legend()

            ax2.grid(
                True,
                alpha=0.3
            )

            st.pyplot(
                fig2,
                clear_figure=True
            )

            # ------------------------------------------------
            # Step 4
            # ------------------------------------------------

            status.info(
                "Step 4/4 — Running machine-learning classification..."
            )

            progress.progress(
                85
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = make_prediction(
                features
            )

            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            if result["status"] == "success":

                display_result(
                    result
                )

                # --------------------------------------------
                # Model information
                # --------------------------------------------

                with st.expander(
                    "🔎 Model Information"
                ):

                    st.write(
                        "Features used by the trained model:"
                    )

                    st.code(
                        "\n".join(
                            result[
                                "model_features"
                            ]
                        )
                    )

            elif result["status"] == "no_model":

                st.error(
                    """
                    ❌ Trained ML model not found.
                    """
                )

                st.code(
                    f"Expected model:\n{MODEL_PATH}"
                )

                st.info(
                    """
                    Train your model first and save it
                    at the location above.
                    """
                )

            elif result["status"] == "feature_error":

                st.error(
                    "❌ Model and application features do not match."
                )

                st.write(
                    "Missing features:"
                )

                st.code(
                    "\n".join(
                        result["missing"]
                    )
                )

                st.write(
                    "Features expected by model:"
                )

                st.code(
                    "\n".join(
                        result["expected"]
                    )
                )

                st.warning(
                    """
                    Retrain the model using the same
                    feature definitions used by this app.
                    """
                )

            elif result["status"] == "prediction_error":

                st.error(
                    "❌ Prediction failed."
                )

                st.code(
                    result["message"]
                )

            elif result["status"] == "error":

                st.error(
                    "❌ Could not load the ML model."
                )

                st.code(
                    result["message"]
                )

            # ------------------------------------------------
            # Extracted data
            # ------------------------------------------------

            with st.expander(
                "📋 View Extracted Movement Data"
            ):

                st.dataframe(
                    data,
                    use_container_width=True
                )

            # ------------------------------------------------
            # Extracted feature values
            # ------------------------------------------------

            with st.expander(
                "🧮 View Calculated Features"
            ):

                feature_df = pd.DataFrame(
                    {
                        "Feature": features.keys(),
                        "Value": features.values()
                    }
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # Download movement data
            # ------------------------------------------------

            csv_data = data.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Movement Data",
                data=csv_data,
                file_name="movement_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

            progress.progress(
                100
            )

            status.success(
                "✅ Analysis completed successfully."
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    try:

        os.unlink(
            video_path
        )

    except Exception:

        pass


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    """
    Early Parkinson Movement Screening • MediaPipe + Machine Learning
    """
)