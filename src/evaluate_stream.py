"""End-to-end evaluation of the streaming consumer path on the injected demo stream.

The producer now streams ``data/processed/dataset_with_anomalies.csv`` (the
anomaly-injected dataset) in chronological order, sending only raw order
fields. The consumer computes every model feature LIVE from its genuine
running state (seeded from the pre-injection baseline exactly like
production, src/consumer.py) and never sees ``is_anomaly`` / ``anomaly_type``
or any precomputed feature column.

This script replays that exact stream through the same code path
(``LiveFeatureState.compute_features`` -> consumer scaling -> LOF ->
threshold), flags rows using the recalibrated anomaly threshold (default: the
consumer-path score percentile at the true anomaly rate, ~3.98), and
evaluates the flags against the TRUE labels, which are known only to this
evaluation, not to the payload.

Reports:
* consumer-path precision / recall / F1 and flag rate on the full stream
* recall by anomaly type (price_spike, delivery_delay, inventory_depletion)
* FP rate on the ~94.5% normal population
* native-space reference (the model scored on the training CSV's own feature
  columns at the same threshold) so the demo number can be compared to the
  ceiling the model was fit on.

Run:
  python -m src.evaluate_stream                    # full ~192k-row stream
  python -m src.evaluate_stream --stream-size 3000 # smoke test
  python -m src.evaluate_stream --threshold 4.0    # override threshold
  python -m src.evaluate_stream --shuffle-window-days 7  # near-chronological
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.evaluate_realistic import scale_matrix
from src.live_state import FEATURE_COLUMNS, LiveFeatureState
from src.model_registry import resolve_model_path
from src.offline_reference import build_offline_reference as _build_offline_reference
from src.producer import RAW_FIELD_COLUMNS, build_stream

CSV_PATH = Path("data/processed/dataset_with_anomalies.csv")
RAW_PATH = Path("data/raw/DataCoSupplyChainDataset.csv")
SCALER_PATH = Path("models/lof_mad_price_scaler.joblib")
ORDER_ITEM_ID = "Order Item Id"
RANDOM_STATE = 42
ANOMALY_TYPES = ["price_spike", "delivery_delay", "inventory_depletion"]
LINE = "=" * 96
PROGRESS_LOG = Path("data/eval_progress.log")

def _log(msg: str) -> None:
    """Append a timestamped progress line to a file (survives timeouts/kills)."""
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.perf_counter():.0f} {msg}\n")

def consumer_path_recalibrated_threshold(model, scaler, injected: pd.DataFrame) -> float:
    """Recompute the consumer-path recalibrated threshold (the ~3.98 default).

    Mirrors src/evaluate_realistic.py: score the model on the consumer's raw
    (pre-injection) feature path over the clean baseline, take the score
    percentile at 1 - true_anomaly_rate of the normal TRAINING rows. Values
    are aligned back to their rows by Order Item Id.
    """
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
    scores_raw = -model.score_samples(scale_matrix(X_raw, scaler))
    score_by_id = pd.Series(scores_raw, index=ref.index)  

    y = injected["is_anomaly"].astype(int).to_numpy()
    normal_idx = np.flatnonzero(y == 0)
    train_norm, _ = train_test_split(
        normal_idx, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    train_ids = set(injected.loc[train_norm, ORDER_ITEM_ID].astype(str))
    true_rate = float(y.mean())
    train_scores = score_by_id.loc[score_by_id.index.isin(train_ids)]
    return float(np.percentile(train_scores, 100 * (1 - true_rate)))

def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f)

def per_type_recall(order_ids: list[str], flags: np.ndarray, types_map: dict[str, str]) -> None:
    print("  recall by anomaly type:")
    for anomaly_type in ANOMALY_TYPES:
        mask = np.array([types_map.get(oid) == anomaly_type for oid in order_ids])
        total = int(mask.sum())
        tp = int((mask & flags).sum()) if total else 0
        recall = tp / total if total else 0.0
        print(f"    {anomaly_type:<22} recall={recall:.4f}  ({tp}/{total} true positives)")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-size", type=int, default=None, help="replay only the first N stream orders")
    parser.add_argument("--threshold", type=float, default=None, help="override the recalibrated threshold")
    parser.add_argument(
        "--shuffle-window-days",
        type=int,
        default=0,
        help="shuffle within rolling N-day blocks instead of pure chronological",
    )
    args = parser.parse_args()

    PROGRESS_LOG.unlink(missing_ok=True)
    _log("start")
    model = joblib.load(resolve_model_path())
    scaler = joblib.load(SCALER_PATH)
    _log("model loaded")

    injected = pd.read_csv(CSV_PATH, encoding="latin1")
    injected[ORDER_ITEM_ID] = injected[ORDER_ITEM_ID].astype(str)
    labels = injected.set_index(ORDER_ITEM_ID)["is_anomaly"].astype(int)
    types_map = injected.set_index(ORDER_ITEM_ID)["anomaly_type"].astype(str).to_dict()
    true_rate = float(labels.mean())

    threshold = args.threshold
    if threshold is None:
        _log("calibration")
        threshold = consumer_path_recalibrated_threshold(model, scaler, injected)
        _log("calibration done")
    print(LINE)
    print(f"true anomaly rate in injected dataset: {true_rate:.4f}")
    print(f"recalibrated threshold (consumer path): {threshold:.4f}")
    print(LINE)

    _log("build_stream")
    stream = build_stream(
        data_path=CSV_PATH,
        shuffle_window_days=args.shuffle_window_days,
        max_rows=args.stream_size,
    )
    _log(f"stream rows: {len(stream)}")
    print(f"stream rows: {len(stream)} (order: {'rolling %d-day shuffle' % args.shuffle_window_days if args.shuffle_window_days else 'chronological'})")

    raw = pd.read_csv(RAW_PATH, encoding="latin1")
    exclude = set(stream[ORDER_ITEM_ID].astype(str)) & set(raw[ORDER_ITEM_ID].astype(str))
    _log(f"seed (exclude {len(exclude)} ids)")
    live_state = LiveFeatureState()
    live_state.seed_from_historical(raw, exclude_order_item_ids=exclude)
    _log("seed done")
    print(f"excluded {len(exclude)} stream order ids from the causal seed")

    order_ids: list[str] = []
    feature_maps: list[dict[str, float]] = []
    raw_columns = [c for c in RAW_FIELD_COLUMNS if c in stream.columns]
    _log("replay")
    for record in stream[raw_columns].to_dict("records"):
        feature_maps.append(live_state.compute_features(record))
        order_ids.append(str(record[ORDER_ITEM_ID]))
    _log("replay done")

    X_stream = np.array([[fm[c] for c in FEATURE_COLUMNS] for fm in feature_maps], dtype=float)
    scores_stream = -model.score_samples(scale_matrix(X_stream, scaler))
    flags_stream = scores_stream >= threshold

    y_stream = np.array([int(labels.get(oid, 0)) for oid in order_ids], dtype=int)
    assert len(y_stream) == len(flags_stream) and (y_stream.sum() > 0), "labels missing for stream rows"

    print(f"\n{'=' * 96}\nA. CONSUMER PATH (live features, recalibrated threshold {threshold:.4f})\n{'=' * 96}")
    p, r, f = prf(y_stream, flags_stream.astype(int))
    normals = y_stream == 0
    fp_rate = float((flags_stream & normals).sum() / normals.sum()) if normals.sum() else 0.0
    print(f"flag rate      = {flags_stream.mean():.2%}  (true rate {true_rate:.2%})")
    print(f"precision      = {p:.4f}")
    print(f"recall         = {r:.4f}")
    print(f"F1             = {f:.4f}")
    print(f"FP rate        = {fp_rate:.2%} of {int(normals.sum())} normals")
    per_type_recall(order_ids, flags_stream, types_map)

    print(f"\n{'=' * 96}\nB. NATIVE-SPACE REFERENCE (training CSV features, same threshold)\n{'=' * 96}")
    X_native = injected[FEATURE_COLUMNS].to_numpy(dtype=float)
    scores_native = -model.score_samples(scale_matrix(X_native, scaler))
    flags_native = scores_native >= threshold
    y_native = injected["is_anomaly"].astype(int).to_numpy()
    p, r, f = prf(y_native, flags_native.astype(int))
    print(f"flag rate      = {flags_native.mean():.2%}")
    print(f"precision      = {p:.4f}")
    print(f"recall         = {r:.4f}")
    print(f"F1             = {f:.4f}")
    print("  recall by anomaly type:")
    for anomaly_type in ANOMALY_TYPES:
        mask = injected["anomaly_type"] == anomaly_type
        total = int(mask.sum())
        tp = int((mask.to_numpy() & flags_native).sum())
        print(f"    {anomaly_type:<22} recall={tp / total if total else 0.0:.4f}  ({tp}/{total})")

    print(LINE)

if __name__ == "__main__":
    main()
