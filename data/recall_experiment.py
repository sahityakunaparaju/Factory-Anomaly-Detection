"""One-off experiment (scratch): deployed-LOF vs full-data-LOF on the consumer path.

Replays the full injected stream through LiveFeatureState exactly like
evaluate_stream does, then for each model (deployed vs LOF fit on ALL rows):
- calibrates the consumer-path threshold at the true anomaly rate
- sweeps alternative thresholds with per-type recall / precision / F1
- prints per-type score distributions (median / p95) vs normals

Scores and labels are saved to data/recall_experiment.npz for later use.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from src.evaluate_realistic import scale_matrix
from src.live_state import FEATURE_COLUMNS, LiveFeatureState
from src.model_registry import resolve_model_path
from src.offline_reference import build_offline_reference
from src.producer import RAW_FIELD_COLUMNS, build_stream

CSV = "data/processed/dataset_with_anomalies.csv"
RAW = "data/raw/DataCoSupplyChainDataset.csv"
SCALER = "models/lof_mad_price_scaler.joblib"
OUT = Path("data/recall_experiment.npz")

ORDER_ITEM_ID = "Order Item Id"
RANDOM_STATE = 42
ANOMALY_TYPES = ["price_spike", "delivery_delay", "inventory_depletion"]

def calibrate(model, scaler, injected: pd.DataFrame) -> float:
    """Consumer-path threshold: score percentile of training normals at 1 - true rate."""
    raw = pd.read_csv(RAW, encoding="latin1")
    raw[ORDER_ITEM_ID] = raw[ORDER_ITEM_ID].astype(str)
    ref = build_offline_reference(raw)
    X_raw = np.column_stack(
        [
            ref["ref_price_deviation"],
            ref["delivery_delay_deviation"],
            ref["discount_rate_anomaly"],
            ref["stock_after_order"],
            ref["days_of_cover_remaining"],
        ]
    ).astype(float)
    scores = -model.score_samples(scale_matrix(X_raw, scaler))
    score_by_id = pd.Series(scores, index=ref.index)
    y = injected["is_anomaly"].astype(int).to_numpy()
    normal_idx = np.flatnonzero(y == 0)
    train_norm, _ = train_test_split(
        normal_idx, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    train_ids = set(injected.loc[train_norm, ORDER_ITEM_ID].astype(str))
    true_rate = float(y.mean())
    return float(np.percentile(score_by_id.loc[score_by_id.index.isin(train_ids)], 100 * (1 - true_rate)))

def report(name: str, model, scaler, X_stream, y, types_map, order_ids) -> None:
    scores = -model.score_samples(scale_matrix(X_stream, scaler))
    print(f"\n=== {name} ===")
    th = calibrate(model, scaler, injected_df)
    print(f"recalibrated threshold: {th:.4f}")
    for t in [round(th, 4), 3.0, 4.0, 6.0, 9.0]:
        flags = scores >= t
        p, r, f, _ = precision_recall_fscore_support(
            y, flags.astype(int), average="binary", zero_division=0
        )
        normals = y == 0
        fp = float((flags & normals).sum() / normals.sum()) if normals.sum() else 0.0
        print(f"  th={t:<6} flag={flags.mean():.2%} prec={p:.4f} rec={r:.4f} F1={f:.4f} FP={fp:.2%}", end="")
        for at in ANOMALY_TYPES:
            mask = np.array([types_map.get(o) == at for o in order_ids])
            tot = int(mask.sum())
            tp = int((mask & flags).sum()) if tot else 0
            print(f" | {at}={tp/tot:.1%}" if tot else f" | {at}=n/a", end="")
        print()
    print("  score distributions (median / p95):")
    for at in ANOMALY_TYPES + ["normal"]:
        if at == "normal":
            mask = y == 0
        else:
            mask = np.array([types_map.get(o) == at for o in order_ids])
        s = scores[mask]
        print(f"    {at:<20} med={np.median(s):.3f} p95={np.percentile(s, 95):.3f} (n={int(mask.sum())})")
    return scores

start = time.perf_counter()
injected_df = pd.read_csv(CSV, encoding="latin1")
injected_df[ORDER_ITEM_ID] = injected_df[ORDER_ITEM_ID].astype(str)
labels = injected_df.set_index(ORDER_ITEM_ID)["is_anomaly"].astype(int)
types_map = injected_df.set_index(ORDER_ITEM_ID)["anomaly_type"].astype(str).to_dict()

stream = build_stream(data_path=CSV, max_rows=None)
order_ids = stream[ORDER_ITEM_ID].astype(str).tolist()
y = np.array([int(labels.get(o, 0)) for o in order_ids], dtype=int)

raw = pd.read_csv(RAW, encoding="latin1")
exclude = set(stream[ORDER_ITEM_ID].astype(str)) & set(raw[ORDER_ITEM_ID].astype(str))
ls = LiveFeatureState()
ls.seed_from_historical(raw, exclude_order_item_ids=exclude)
print(f"seeded ({len(exclude)} excluded); replaying {len(stream)} rows...")
raw_columns = [c for c in RAW_FIELD_COLUMNS if c in stream.columns]
X_stream = np.array(
    [list(ls.compute_features(rec).values()) for rec in stream[raw_columns].to_dict("records")],
    dtype=float,
)
np.savez(OUT, X=X_stream, y=y, order_ids=np.array(order_ids))
print(f"replay done in {time.perf_counter()-start:.0f}s; saved {OUT}")

scaler = joblib.load(SCALER)
deployed = joblib.load(resolve_model_path())
report("DEPLOYED (normal-only train)", deployed, scaler, X_stream, y, types_map, order_ids)

X_all = injected_df[FEATURE_COLUMNS].to_numpy(dtype=float)
X_all = np.nan_to_num(X_all, nan=0.0)
X_all_s = scale_matrix(X_all, scaler)
full_lof = LocalOutlierFactor(contamination=0.065, novelty=True, n_neighbors=20, n_jobs=-1)
t0 = time.perf_counter()
full_lof.fit(X_all_s)
print(f"\nfull-data LOF fit: {time.perf_counter()-t0:.0f}s (on {len(X_all_s)} rows)")
report("FULL-DATA TRAIN LOF", full_lof, scaler, X_stream, y, types_map, order_ids)
print(f"\ntotal {time.perf_counter()-start:.0f}s")
