"""Evaluate the deployed LOF model on a realistic (population-proportioned)
holdout and diagnose / recalibrate its anomaly decision threshold.

Two scoring paths are examined:

* Native  - the model scored on the training CSV's own feature columns (the
            space it was fit in).
* Consumer - the model scored on the raw-based features the streaming consumer
             actually produces (pre-injection baseline), i.e. what the demo
             stream really scores.

For each path we report flag rates per segment, precision/recall/F1 on
realistic-distribution holdouts (not the balanced 50/50 test), and the effect
of recalibrating the decision threshold to the true anomaly rate via the score
percentile of training rows.

Run:  python -m src.evaluate_realistic
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.model_registry import resolve_model_path
from src.offline_reference import build_offline_reference as _build_offline_reference

CSV_PATH = Path("data/processed/dataset_with_anomalies.csv")
RAW_PATH = Path("data/raw/DataCoSupplyChainDataset.csv")
SCALER_PATH = Path("models/lof_mad_price_scaler.joblib")
ORDER_ITEM_ID = "Order Item Id"

FEATURES = [
    "price_deviation_from_supplier_category_avg",
    "delivery_delay_deviation",
    "discount_rate_anomaly",
    "stock_after_order",
    "days_of_cover_remaining",
]
RANDOM_STATE = 42
LINE = "=" * 96

def scale_matrix(X: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    """Replicate the consumer's scaling: MAD-scaled price + standard-scaled rest."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    out = X.copy()
    price_index = int(scaler["price_index"])

    price_scale = scaler.get("price_scale")
    if price_scale is None:
        price_scale = 1.4826 * float(scaler["price_mad"])
    price_scale = float(price_scale)
    if price_scale == 0 or not np.isfinite(price_scale):
        price_scale = 1.0
    out[:, price_index] = (X[:, price_index] - float(scaler["price_median"])) / price_scale
    non_price = [i for i in range(X.shape[1]) if i != price_index]
    out[:, non_price] = scaler["non_price_scaler"].transform(X[:, non_price])
    return out

def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return float(p), float(r), float(f)

def report(label: str, scores: np.ndarray, y: np.ndarray, threshold: float) -> None:
    pred = (scores >= threshold).astype(int)
    p, r, f = prf(y, pred)
    print(
        f"{label:<62} flag={pred.mean():6.2%}  prec={p:.4f}  rec={r:.4f}  f1={f:.4f}  (n={len(y)})"
    )

def main() -> None:
    csv_df = pd.read_csv(CSV_PATH, encoding="latin1")
    csv_df[ORDER_ITEM_ID] = csv_df[ORDER_ITEM_ID].astype(str)
    y = csv_df["is_anomaly"].astype(int).to_numpy()
    X = csv_df[FEATURES].to_numpy(dtype=float)

    model = joblib.load(resolve_model_path())
    scaler = joblib.load(SCALER_PATH)
    Xs = scale_matrix(X, scaler)
    scores = -model.score_samples(Xs)  

    normal_idx = np.flatnonzero(y == 0)
    train_norm, held_norm = train_test_split(
        normal_idx, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    anom_idx = np.flatnonzero(y == 1)
    rng = np.random.default_rng(RANDOM_STATE)
    balanced_norm = rng.choice(held_norm, size=len(anom_idx), replace=False)

    true_rate = float(y.mean())
    cur_thresh = -float(model.offset_)

    train_percentile_thresh = float(np.percentile(scores[train_norm], 100 * (1 - true_rate)))

    print(LINE)
    print("Deployed model: LOF (novelty)  |  contamination=0.065  |  true anomaly rate=%.4f" % true_rate)
    print(f"current threshold (-offset_)={cur_thresh:.4f} | train-percentile threshold={train_percentile_thresh:.4f}")
    print(LINE)

    print("\n--- A. NATIVE scoring (training CSV features: the space the model was fit in) ---")
    print(f"flag rate  train normals={np.mean(scores[train_norm] >= cur_thresh):.2%} | "
          f"held-out normals={np.mean(scores[held_norm] >= cur_thresh):.2%} | "
          f"anomalies={np.mean(scores[anom_idx] >= cur_thresh):.2%} | "
          f"FULL population={np.mean(scores >= cur_thresh):.2%} (true rate {true_rate:.2%})")

    hold_all = np.concatenate([held_norm, anom_idx])
    n_anom_pop = int(round(len(held_norm) * true_rate / (1 - true_rate)))
    anom_pop = rng.choice(anom_idx, size=n_anom_pop, replace=False)
    hold_pop = np.concatenate([held_norm, anom_pop])
    balanced = np.concatenate([balanced_norm, anom_idx])

    print("\n--- realistic-distribution holdouts (native features, current threshold) ---")
    report(f"held-out, all anomalies + all held normals   ({y[hold_all].mean():.1%} anom)",
           scores[hold_all], y[hold_all], cur_thresh)
    report(f"held-out, population-proportioned            ({y[hold_pop].mean():.1%} anom)",
           scores[hold_pop], y[hold_pop], cur_thresh)
    report("balanced test (reference, 50/50)", scores[balanced], y[balanced], cur_thresh)

    print("\n--- B. CONSUMER-PATH scoring (raw-based features the demo stream produces) ---")
    raw = pd.read_csv(RAW_PATH, encoding="latin1")
    raw[ORDER_ITEM_ID] = raw[ORDER_ITEM_ID].astype(str)
    ref = _build_offline_reference(raw)
    X_raw = np.column_stack(
        [
            ref["ref_price_deviation"],
            ref["delivery_delay_deviation"],
            ref["discount_rate_anomaly"],
            ref["stock_after_order"],
            ref["days_of_cover_remaining"],
        ]
    ).astype(float)
    Xs_raw = scale_matrix(X_raw, scaler)
    scores_raw = -model.score_samples(Xs_raw)
    labels = raw[ORDER_ITEM_ID].map(csv_df.set_index(ORDER_ITEM_ID)["is_anomaly"]).to_numpy(dtype=float)
    has_label = ~np.isnan(labels)

    train_ids = set(csv_df.loc[train_norm, ORDER_ITEM_ID])
    train_mask = raw[ORDER_ITEM_ID].isin(train_ids).to_numpy()
    consumer_recal_thresh = float(np.percentile(scores_raw[train_mask], 100 * (1 - true_rate)))

    print(
        f"raw rows scored: {len(scores_raw)} (clean historical rows; the stream contains "
        f"none of the injected anomalies)\n"
        f"flag rate on clean stream, current threshold = {np.mean(scores_raw >= cur_thresh):.2%}\n"
        f"flag rate on clean stream, consumer-recalibrated = {np.mean(scores_raw >= consumer_recal_thresh):.2%}"
    )
    print(
        f"median anomaly score shift on shared rows: "
        f"native={np.median(scores[normal_idx]):.3f} vs consumer-path={np.median(scores_raw):.3f}"
    )
    report("consumer path, current threshold (labeled rows)", scores_raw[has_label],
           labels[has_label].astype(int), cur_thresh)
    report("consumer path, consumer-recalibrated (labeled rows)", scores_raw[has_label],
           labels[has_label].astype(int), consumer_recal_thresh)

    print("\n--- C. RECALIBRATION comparison (before / after) ---")
    print(f"recalibrated threshold (native train percentile)      = {train_percentile_thresh:.4f}")
    print(f"recalibrated threshold (consumer-path train percentile) = {consumer_recal_thresh:.4f}")
    print(
        f"flag rate on FULL population: native-features before={np.mean(scores >= cur_thresh):.2%} "
        f"-> after={np.mean(scores >= train_percentile_thresh):.2%}"
    )
    print(
        f"flag rate on clean stream   : consumer-path   before={np.mean(scores_raw >= cur_thresh):.2%} "
        f"-> after={np.mean(scores_raw >= consumer_recal_thresh):.2%}"
    )
    print("\nrealistic holdout, population-proportioned:")
    report("before recalibration", scores[hold_pop], y[hold_pop], cur_thresh)
    report("after  recalibration", scores[hold_pop], y[hold_pop], train_percentile_thresh)

if __name__ == "__main__":
    main()
