"""
AI-Based MCU Fuse Map Integrity Auditor
---------------------------------------
Train and compare Logistic Regression and Random Forest classifiers.

Inputs:
    data/mcu_fuse_map_audit.csv
    data/mcu_fuse_schema.csv
    src/fuse_decoder.py

Target:
    has_error

Excluded from features:
    die_id, error_type, fuse_map_hex, has_error

Outputs:
    models/best_fuse_error_model.joblib
    models/feature_metadata.json
    models/model_comparison.csv
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "mcu_fuse_map_audit.csv"
SCHEMA_PATH = PROJECT_ROOT / "data" / "mcu_fuse_schema.csv"
SRC_PATH = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "has_error"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Expected 10 fields from mcu_fuse_schema.csv.
EXPECTED_FUSE_FIELDS = [
    "BG_TRIM",
    "HSI_TRIM",
    "LDO_TRIM",
    "TS_CAL1",
    "FLASH_SIZE",
    "PERIPH_EN",
    "TEMP_GRADE",
    "DEBUG_LOCK",
    "RDP",
    "RESERVED",
]

# Only decoded fields and derivative are allowed as model inputs.
IDENTIFIER_COLUMNS = ["die_id"]
LEAKAGE_COLUMNS = ["error_type", "fuse_map_hex", TARGET_COLUMN]


# ---------------------------------------------------------------------
# Import the existing decoder
# ---------------------------------------------------------------------

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from fuse_decoder import decode_fuse_map, load_schema
except ImportError as exc:
    raise ImportError(
        "Could not import decode_fuse_map and load_schema from "
        f"{SRC_PATH / 'fuse_decoder.py'}. Check that the file exists "
        "and exports both functions."
    ) from exc


# ---------------------------------------------------------------------
# Schema and decoder compatibility helpers
# ---------------------------------------------------------------------

def load_project_schema():
    """Load the existing schema using the decoder's available API."""
    signature = inspect.signature(load_schema)
    parameters = signature.parameters

    # Support common schema-loader signatures without changing the decoder.
    for name in ("schema_path", "path", "csv_path", "file_path"):
        if name in parameters:
            return load_schema(**{name: str(SCHEMA_PATH)})

    required_parameters = [
        p
        for p in parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    if not required_parameters:
        return load_schema()

    return load_schema(str(SCHEMA_PATH))


def decode_one_fuse_map(fuse_hex, derivative, schema):
    """
    Call the existing decoder.

    Supports common function signatures:
        decode_fuse_map(hex_value, schema)
        decode_fuse_map(hex_value, derivative, schema)
        decode_fuse_map(hex_value)
    """
    signature = inspect.signature(decode_fuse_map)
    parameters = signature.parameters
    kwargs = {}

    for name, parameter in parameters.items():
        normalized = name.lower()

        if normalized in {
            "fuse_map_hex",
            "fuse_hex",
            "hex_value",
            "hex_string",
            "hex_map",
            "raw_hex",
            "value",
        }:
            kwargs[name] = fuse_hex

        elif normalized in {"derivative", "mcu_derivative", "device"}:
            kwargs[name] = derivative

        elif normalized in {"schema", "fuse_schema", "schema_df"}:
            kwargs[name] = schema

        elif normalized in {
            "schema_path",
            "path",
            "csv_path",
            "file_path",
        }:
            kwargs[name] = str(SCHEMA_PATH)

    required_parameters = [
        p
        for p in parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    if all(p.name in kwargs for p in required_parameters):
        result = decode_fuse_map(**kwargs)
    else:
        # Fallback for decoders with different parameter names.
        candidates = [
            (fuse_hex, derivative, schema),
            (fuse_hex, schema, derivative),
            (fuse_hex, schema),
            (fuse_hex, derivative),
            (fuse_hex,),
        ]

        last_error = None
        result = None

        for args in candidates:
            try:
                signature.bind(*args)
                result = decode_fuse_map(*args)
                break
            except TypeError as exc:
                last_error = exc

        if result is None:
            raise TypeError(
                "Could not match decode_fuse_map's function signature. "
                "Check its parameters in src/fuse_decoder.py."
            ) from last_error

    # Accommodate decoders that return (decoded_fields, other_information).
    if isinstance(result, tuple):
        result = next(
            (
                item
                for item in result
                if isinstance(item, (dict, pd.Series))
            ),
            result[0] if result else None,
        )

    if isinstance(result, pd.Series):
        result = result.to_dict()

    if not isinstance(result, dict):
        raise TypeError(
            "decode_fuse_map must return a dictionary or pandas Series "
            "of decoded fuse fields. Check src/fuse_decoder.py."
        )

    return result


def normalize_decoded_fields(decoded):
    """Normalize decoder field names and verify the expected fields."""
    normalized = {
        str(key).strip().upper(): value
        for key, value in decoded.items()
    }

    missing = [
        field for field in EXPECTED_FUSE_FIELDS
        if field not in normalized
    ]

    if missing:
        raise ValueError(
            "The decoder did not return all 10 expected fields. "
            f"Missing: {missing}. Returned fields: {list(decoded.keys())}. "
            "Check the decoder output and mcu_fuse_schema.csv."
        )

    output = {}

    for field in EXPECTED_FUSE_FIELDS:
        value = normalized[field]

        # Reject missing/non-numeric decoded values instead of silently
        # replacing a failed decode with zero.
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Decoded field {field} has a non-numeric value: {value!r}"
            ) from exc

        if not np.isfinite(number):
            raise ValueError(
                f"Decoded field {field} is not finite: {value!r}"
            )

        output[field] = number

    return output


# ---------------------------------------------------------------------
# Data loading and feature extraction
# ---------------------------------------------------------------------

def load_and_decode_dataset():
    """Read the CSV, decode all fuse maps, and create model features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = {
        "die_id",
        "derivative",
        "fuse_map_hex",
        "has_error",
        "error_type",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("The dataset contains no records.")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The has_error target contains missing values.")

    # Handle boolean, integer, and common text representations.
    if data[TARGET_COLUMN].dtype == object:
        target_values = (
            data[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        target_mapping = {
            "1": 1,
            "true": 1,
            "yes": 1,
            "0": 0,
            "false": 0,
            "no": 0,
        }
        mapped = target_values.map(target_mapping)

        if mapped.isna().any():
            bad_values = target_values[mapped.isna()].unique().tolist()
            raise ValueError(
                f"Unrecognized has_error values: {bad_values}"
            )

        y = mapped.astype(int)
    else:
        y = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "has_error must contain both binary classes 0 and 1."
        )

    schema = load_project_schema()

    decoded_rows = []
    total = len(data)

    print(f"\nDataset records: {total}")
    print("Decoding fuse maps into the 10 schema fields...")

    for row_number, row in data.iterrows():
        try:
            decoded = decode_one_fuse_map(
                fuse_hex=row["fuse_map_hex"],
                derivative=row["derivative"],
                schema=schema,
            )
            decoded_rows.append(normalize_decoded_fields(decoded))

        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode dataset row {row_number}, "
                f"die_id={row.get('die_id', 'unknown')!r}: {exc}"
            ) from exc

    decoded_df = pd.DataFrame(decoded_rows, index=data.index)

    if decoded_df.shape[1] != 10:
        raise AssertionError(
            f"Expected 10 decoded fields, got {decoded_df.shape[1]}."
        )

    # Derivative is a valid categorical input. Keep it separate from
    # the decoded numeric fuse fields.
    derivative = data["derivative"].astype("string").str.strip()

    if derivative.isna().any() or (derivative == "").any():
        raise ValueError("The derivative column contains missing values.")

    X = decoded_df.copy()
    X["derivative"] = derivative.astype(str)

    # Explicit leakage guard: only the 10 decoded fields and derivative
    # may be used as input features.
    forbidden = set(IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS)
    unexpected = forbidden.intersection(X.columns)

    if unexpected:
        raise AssertionError(
            f"Forbidden target/identifier columns in X: {unexpected}"
        )

    print(f"Successfully decoded records: {len(decoded_df)}")
    print(f"Decoded fuse fields: {EXPECTED_FUSE_FIELDS}")
    print(f"Model input features: {list(X.columns)}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index().rename(index={0: "No error", 1: "Error"}))

    return X, y, data, schema


# ---------------------------------------------------------------------
# Model pipelines
# ---------------------------------------------------------------------

def build_models():
    """Create both classifiers with preprocessing inside each pipeline."""
    numeric_features = EXPECTED_FUSE_FIELDS
    categorical_features = ["derivative"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------
# Evaluation and saving
# ---------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    """Print evaluation metrics and return a comparison row."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No error", "Error"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                    Predicted no-error  Predicted error")
    print(f"Actual no-error          {matrix[0, 0]:5d}             {matrix[0, 1]:5d}")
    print(f"Actual error             {matrix[1, 0]:5d}             {matrix[1, 1]:5d}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }


def main():
    warnings.filterwarnings(
        "default",
        category=UserWarning,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, original_data, schema = load_and_decode_dataset()

    # One shared stratified split makes model comparison fair.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/test split")
    print("-" * 40)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")
    print(f"Training error rate: {y_train.mean():.3f}")
    print(f"Testing error rate : {y_test.mean():.3f}")

    models = build_models()
    fitted_models = {}
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        results.append(
            evaluate_model(name, model, X_test, y_test)
        )

    comparison = (
        pd.DataFrame(results)
        .sort_values(
            by=["f1_score", "recall", "precision"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(
        comparison[
            ["model", "accuracy", "precision", "recall", "f1_score"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    # Select by F1-score, with recall as the tie-breaker. This is useful
    # for an auditor where missed faulty records are important.
    best_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_name]

    model_path = MODEL_DIR / "best_fuse_error_model.joblib"
    joblib.dump(best_model, model_path)

    # Save enough metadata to understand and reproduce the feature setup.
    schema_fields = []

    if isinstance(schema, pd.DataFrame):
        schema_fields = schema.to_dict(orient="records")
    elif isinstance(schema, list):
        schema_fields = schema
    elif isinstance(schema, dict):
        schema_fields = schema

    metadata = {
        "project": "AI-Based MCU Fuse Map Integrity Auditor",
        "target": TARGET_COLUMN,
        "target_classes": {
            "0": "No error",
            "1": "Error",
        },
        "selected_model": best_name,
        "selection_criterion": "Highest F1-score, then recall, then precision",
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "comparison_file": str(comparison_path.relative_to(PROJECT_ROOT)),
        "dataset_file": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "schema_file": str(SCHEMA_PATH.relative_to(PROJECT_ROOT)),
        "feature_columns": list(X.columns),
        "decoded_fuse_fields": EXPECTED_FUSE_FIELDS,
        "categorical_features": ["derivative"],
        "excluded_columns": [
            "die_id",
            "error_type",
            "fuse_map_hex",
            "has_error",
        ],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_metrics": comparison.to_dict(orient="records"),
        "schema": schema_fields,
    }

    metadata_path = MODEL_DIR / "feature_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Selected model : {best_name}")
    print(f"Saved model    : {model_path}")
    print(f"Saved metadata : {metadata_path}")
    print(f"Saved metrics  : {comparison_path}")
    print("\nReminder: these are held-out test metrics, not proof of")
    print("production readiness. Validate on independent fuse-map data.")


if __name__ == "__main__":
    main()