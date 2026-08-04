from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.model_registry import MANIFEST_PATH

def resolve_price_scale(train_price: np.ndarray, median: float) -> float:
    """Robust, zero-inflation-aware scale for the price-deviation feature.

    The price-deviation feature is heavily zero-inflated (>=50% exact zeros
    whenever the rolling-median baseline equals the current price), so the
    plain MAD can be exactly 0. The old fallback to 1e-6 turned every $1 of
    real deviation into a scaled value of 1,000,000, swamping the other four
    features and producing absurd reason strings. Instead we fall back to the
    MAD over the NONZERO deviations, then the std, then 1.0 - never a
    near-zero epsilon.
    """
    mad = float(np.median(np.abs(train_price - median)))
    scale = 1.4826 * mad
    if scale == 0 or not np.isfinite(scale):
        nonzero = train_price[np.abs(train_price) > 0]
        if len(nonzero) > 1:
            nonzero_mad = float(np.median(np.abs(nonzero - np.median(nonzero))))
            scale = 1.4826 * nonzero_mad
    if scale == 0 or not np.isfinite(scale):
        scale = float(np.std(train_price))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    return scale

def mad_scale_price_feature(X_train: np.ndarray, X_test: np.ndarray, price_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit a MAD-scale transform on the training price feature only.

    The transform uses the median and MAD of the training data. A degenerate
    (zero-MAD) price column falls back to a data-driven scale via
    :func:`resolve_price_scale` instead of a near-zero epsilon.
    """
    train_price = X_train[:, price_index]
    test_price = X_test[:, price_index]

    median = float(np.median(train_price))
    scale = resolve_price_scale(train_price, median)

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, price_index] = (train_price - median) / scale
    X_test_scaled[:, price_index] = (test_price - median) / scale
    return X_train_scaled, X_test_scaled

DATA_PATH = Path("data/processed/dataset_with_anomalies.csv")
MODEL_DIR = Path("models")
RANDOM_STATE = 42
CONTAMINATION = 0.065
FEATURE_COLUMNS = [
    "price_deviation_from_supplier_category_avg",
    "delivery_delay_deviation",
    "discount_rate_anomaly",
    "stock_after_order",
    "days_of_cover_remaining",
]

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    required = ["is_anomaly", "anomaly_type"] + FEATURE_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return df

def build_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_df = df[df["is_anomaly"] == 0].copy()
    anomaly_df = df[df["is_anomaly"] == 1].copy()

    normal_train_df, normal_eval_df = train_test_split(
        normal_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    normal_eval_sample = normal_eval_df.sample(
        n=min(len(normal_eval_df), len(anomaly_df)),
        random_state=RANDOM_STATE,
    )
    test_df = pd.concat([anomaly_df, normal_eval_sample], ignore_index=True)
    return normal_train_df, test_df

def score_model(model, X_train: np.ndarray, X_test: np.ndarray, y_true: pd.Series) -> dict[str, object]:
    model.fit(X_train)
    pred_labels = model.predict(X_test)
    predicted_anomaly = (pred_labels == -1).astype(int)
    true_anomaly = y_true.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_anomaly,
        predicted_anomaly,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(true_anomaly, predicted_anomaly, labels=[0, 1])

    return {
        "model": model,
        "predicted_anomaly": predicted_anomaly,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }

def evaluate_by_anomaly_type(test_df: pd.DataFrame, predicted_anomaly: np.ndarray) -> pd.DataFrame:
    rows = []
    for anomaly_type in ["price_spike", "delivery_delay", "inventory_depletion"]:
        mask = test_df["anomaly_type"] == anomaly_type
        true = test_df.loc[mask, "is_anomaly"].astype(int).to_numpy()
        predicted = predicted_anomaly[mask]
        tp = int(np.sum((true == 1) & (predicted == 1)))
        total = int(np.sum(true == 1))
        recall = tp / total if total else 0.0
        rows.append(
            {
                "anomaly_type": anomaly_type,
                "true_count": total,
                "true_positives": tp,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows)

def build_models() -> dict[str, object]:
    return {
        "isolation_forest": IsolationForest(
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_estimators=300,
        ),
        "lof": LocalOutlierFactor(
            contamination=CONTAMINATION,
            novelty=True,
            n_neighbors=20,
        ),
    }

def run_model_suite(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: pd.Series,
    test_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, model in build_models().items():
        result = score_model(model, X_train, X_test, y_test)
        result["recall_by_type"] = evaluate_by_anomaly_type(test_df, result["predicted_anomaly"])
        results[name] = result
    return results

def summarize_results_for_comparison(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "model": name,
                "precision": round(float(result["precision"]), 6),
                "recall": round(float(result["recall"]), 6),
                "f1": round(float(result["f1"]), 6),
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    df = load_dataset()
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    train_df, test_df = build_train_test_split(df)

    X_train_raw = train_df[FEATURE_COLUMNS].copy().to_numpy(dtype=float)
    X_test_raw = test_df[FEATURE_COLUMNS].copy().to_numpy(dtype=float)
    y_test = test_df["is_anomaly"].astype(int)

    X_train_raw = np.nan_to_num(X_train_raw, nan=np.nanmedian(X_train_raw, axis=0))
    X_test_raw = np.nan_to_num(X_test_raw, nan=np.nanmedian(X_train_raw, axis=0))

    baseline_results = run_model_suite(X_train_raw, X_test_raw, y_test, test_df)

    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train_standard = scaler.transform(X_train_raw)
    X_test_standard = scaler.transform(X_test_raw)
    standard_results = run_model_suite(X_train_standard, X_test_standard, y_test, test_df)

    price_index = FEATURE_COLUMNS.index("price_deviation_from_supplier_category_avg")
    X_train_mad_price, X_test_mad_price = mad_scale_price_feature(X_train_raw, X_test_raw, price_index=price_index)
    non_price_columns = [idx for idx in range(len(FEATURE_COLUMNS)) if idx != price_index]
    scaler_non_price = StandardScaler()
    scaler_non_price.fit(X_train_raw[:, non_price_columns])
    X_train_mad_price[:, non_price_columns] = scaler_non_price.transform(X_train_raw[:, non_price_columns])
    X_test_mad_price[:, non_price_columns] = scaler_non_price.transform(X_test_raw[:, non_price_columns])

    mad_price_results = run_model_suite(X_train_mad_price, X_test_mad_price, y_test, test_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_name = max(
        mad_price_results,
        key=lambda name: mad_price_results[name]["f1"],
    )
    best_model = mad_price_results[best_name]["model"]
    joblib.dump(best_model, MODEL_DIR / f"{best_name}.joblib")

    price_median = float(np.median(X_train_raw[:, price_index]))

    price_mad = float(np.median(np.abs(X_train_raw[:, price_index] - price_median)))
    price_scale = resolve_price_scale(X_train_raw[:, price_index], price_median)

    scaler_bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "price_index": price_index,
        "price_median": price_median,
        "price_mad": price_mad,
        "price_scale": price_scale,
        "non_price_scaler": scaler_non_price,
    }
    joblib.dump(scaler_bundle, MODEL_DIR / "lof_mad_price_scaler.joblib")

    manifest = {
        "model_name": best_name,
        "model_file": f"{best_name}.joblib",
        "threshold": float(-best_model.offset_),
        "price_scale": price_scale,
        "contamination": CONTAMINATION,
        "f1": float(mad_price_results[best_name]["f1"]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Current-model manifest written to:", MANIFEST_PATH)

    print("Feature columns:", FEATURE_COLUMNS)
    print("Train rows (normal-only):", len(X_train_raw))
    print("Test rows:", len(X_test_raw))
    print("Anomaly rows in test:", int((y_test == 1).sum()))
    print("Normal rows in test:", int((y_test == 0).sum()))
    print("Contamination used:", CONTAMINATION)
    print()

    comparison = pd.DataFrame(
        {
            "model": sorted(set(baseline_results) | set(standard_results) | set(mad_price_results)),
        }
    )
    for name in comparison["model"]:
        comparison.loc[comparison["model"] == name, "precision_before"] = round(
            float(baseline_results[name]["precision"]), 6
        )
        comparison.loc[comparison["model"] == name, "recall_before"] = round(
            float(baseline_results[name]["recall"]), 6
        )
        comparison.loc[comparison["model"] == name, "f1_before"] = round(
            float(baseline_results[name]["f1"]), 6
        )
        comparison.loc[comparison["model"] == name, "precision_standard"] = round(
            float(standard_results[name]["precision"]), 6
        )
        comparison.loc[comparison["model"] == name, "recall_standard"] = round(
            float(standard_results[name]["recall"]), 6
        )
        comparison.loc[comparison["model"] == name, "f1_standard"] = round(
            float(standard_results[name]["f1"]), 6
        )
        comparison.loc[comparison["model"] == name, "precision_mad_price"] = round(
            float(mad_price_results[name]["precision"]), 6
        )
        comparison.loc[comparison["model"] == name, "recall_mad_price"] = round(
            float(mad_price_results[name]["recall"]), 6
        )
        comparison.loc[comparison["model"] == name, "f1_mad_price"] = round(
            float(mad_price_results[name]["f1"]), 6
        )

    print("Before vs StandardScaler vs MAD-price scaling comparison table:")
    print(comparison.to_string(index=False))
    print()

    print("MAD-price scaled evaluation results:")
    for name, result in mad_price_results.items():
        print(f"Model: {name}")
        print("  Precision:", round(float(result["precision"]), 6))
        print("  Recall:", round(float(result["recall"]), 6))
        print("  F1:", round(float(result["f1"]), 6))
        print("  Confusion matrix:")
        print(result["confusion_matrix"])
        print("  Recall by anomaly_type:")
        print(result["recall_by_type"].to_string(index=False))
        print()

    print("Best MAD-scaled model saved to:", MODEL_DIR / f"{best_name}.joblib")

if __name__ == "__main__":
    main()
