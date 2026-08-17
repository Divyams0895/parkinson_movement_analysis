import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/features.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "parkinson_screening_model.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

    "mean_distance",

    "std_distance",

    "movement_range",

    "mean_velocity",

    "std_velocity",

    "tap_count",

    "taps_per_second",

    "movement_variability"
]


TARGET_COLUMN = "label"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"""
Dataset not found.

Expected file:
{DATA_PATH}

Please create:
data/features.csv
"""
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print("\nDataset loaded successfully.")

    print(
        f"Number of samples: {len(df)}"
    )

    print(
        f"Number of columns: {len(df.columns)}"
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    required_columns = (
        FEATURE_COLUMNS +
        [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing columns:\n"
            +
            "\n".join(missing_columns)
        )

    # Remove rows containing missing values

    before = len(df)

    df = df.dropna(
        subset=required_columns
    )

    after = len(df)

    if before != after:

        print(
            f"\nRemoved {before - after} "
            "rows containing missing values."
        )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    print("\nClass distribution:")

    print(
        df[TARGET_COLUMN].value_counts()
    )

    unique_labels = sorted(
        df[TARGET_COLUMN].unique()
    )

    if not set(unique_labels).issubset(
        {0, 1}
    ):

        raise ValueError(
            """
The label column must contain:

0 = Healthy
1 = Parkinsonian-pattern
"""
        )

    if len(unique_labels) < 2:

        raise ValueError(
            """
The dataset contains only one class.

You need both:
0 = Healthy
1 = Parkinsonian-pattern
"""
        )

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=10,

        min_samples_split=4,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    except ValueError:

        auc = 0.0

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Healthy",
                "Parkinsonian-pattern"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(model):

    importance = pd.DataFrame({

        "Feature":
            FEATURE_COLUMNS,

        "Importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n")
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n")
    print("=" * 60)

    print(
        f"Model saved successfully:"
    )

    print(
        MODEL_PATH
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print(
        "PARKINSON MOVEMENT SCREENING MODEL TRAINING"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    df = validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        TARGET_COLUMN
    ]

    print("\nFeatures used:")

    for feature in FEATURE_COLUMNS:

        print(
            f"  • {feature}"
        )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print("\n")
    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = train_model(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = show_feature_importance(
        model
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    # --------------------------------------------------------
    # Save evaluation report
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    report_path = (
        "results/model_evaluation.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(
            "PARKINSON MOVEMENT SCREENING\n"
        )

        file.write(
            "MODEL EVALUATION REPORT\n"
        )

        file.write(
            "=" * 60
        )

        file.write("\n\n")

        file.write(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision: "
            f"{metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall: "
            f"{metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1 Score: "
            f"{metrics['f1']:.4f}\n"
        )

        file.write(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}\n"
        )

        file.write("\n\n")

        file.write(
            "FEATURE IMPORTANCE\n"
        )

        file.write(
            importance.to_string(
                index=False
            )
        )

        file.write("\n\n")

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            str(
                metrics[
                    "confusion_matrix"
                ]
            )
        )

    print(
        f"\nEvaluation report saved to:"
    )

    print(
        report_path
    )

    print("\nTraining process completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()