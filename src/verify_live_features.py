"""Verify the consumer's LIVE feature computation against the offline pipeline.

Two modes
---------
* Chronological (default): the earlier rows become the historical seed and the
  LAST N rows are replayed in chronological order through the consumer's live
  state. Stream rows are excluded from the causal seeds (price windows,
  inventory consumption/demand), so the state at each stream order is exactly
  what the offline pipeline had when it processed that order. This proves the
  live state logic is bit-exact against the offline pipeline.

* Shuffled (``--shuffle``): rows are replayed in the producer's shuffled order
  (``df.sample(frac=1.0, random_state=42)``) with the FULL dataset seeded, just
  like production. This quantifies how much the live features diverge from the
  offline reference when orders arrive out of chronological order, and whether
  that divergence is large enough to flip the model's anomaly verdict for a
  row.

Run:
  python -m src.verify_live_features --stream-size 150          # chronological
  python -m src.verify_live_features --shuffle --stream-size 50000  # shuffled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.consumer import _prepare_features
from src.live_state import LiveFeatureState
from src.model_registry import resolve_model_path
from src.offline_reference import build_offline_reference as _build_offline_reference

RAW_DATA_PATH = Path("data/raw/DataCoSupplyChainDataset.csv")
TRAINED_CSV_PATH = Path("data/processed/dataset_with_anomalies.csv")
SCALER_PATH = Path("models/lof_mad_price_scaler.joblib")
ORDER_ITEM_ID = "Order Item Id"
ORDER_DATE_COLUMN = "order date (DateOrders)"

FEATURES = [
    "price_deviation_from_supplier_category_avg",
    "delivery_delay_deviation",
    "stock_after_order",
    "days_of_cover_remaining",
]
ALL_FEATURES = FEATURES + ["discount_rate_anomaly"]

REF_COLUMN = {
    "price_deviation_from_supplier_category_avg": "ref_price_deviation",
    "delivery_delay_deviation": "delivery_delay_deviation",
    "discount_rate_anomaly": "discount_rate_anomaly",
    "stock_after_order": "stock_after_order",
    "days_of_cover_remaining": "days_of_cover_remaining",
}

TOL = 1e-6

def _close(a: float, b: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=TOL, atol=TOL))

def _replay(live_state: LiveFeatureState, stream: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Replay the stream through compute_features(); return per-order live/offline rows."""
    rows: list[dict[str, object]] = []
    for _, order in stream.iterrows():
        live = live_state.compute_features(order.to_dict())
        order_ref = ref.loc[str(order[ORDER_ITEM_ID])]
        row: dict[str, object] = {
            ORDER_ITEM_ID: str(order[ORDER_ITEM_ID]),
            "order_date": str(order[ORDER_DATE_COLUMN].date()),
            "region": order["Order Region"],
            "category": order["Category Name"],
            "product": order["Product Card Id"],
        }
        for feature in ALL_FEATURES:
            live_value = float(live[feature])
            offline_value = float(order_ref[REF_COLUMN[feature]])
            row[f"{feature}__live"] = live_value
            row[f"{feature}__offline"] = offline_value
            row[f"{feature}__delta"] = (
                live_value - offline_value
                if not (np.isnan(live_value) or np.isnan(offline_value))
                else float("nan")
            )
            row[f"{feature}__match"] = _close(live_value, offline_value)
        rows.append(row)
    return pd.DataFrame(rows)

def _flip_analysis(table: pd.DataFrame) -> None:
    """Compare the model verdict on live-shuffled vs offline features per row."""
    model = joblib.load(resolve_model_path())
    scaler_bundle = joblib.load(SCALER_PATH)

    live_maps = [
        {feature: row[f"{feature}__live"] for feature in ALL_FEATURES}
        for _, row in table.iterrows()
    ]
    offline_maps = [
        {feature: row[f"{feature}__offline"] for feature in ALL_FEATURES}
        for _, row in table.iterrows()
    ]
    X_live = np.vstack([_prepare_features(m, scaler_bundle) for m in live_maps])
    X_offline = np.vstack([_prepare_features(m, scaler_bundle) for m in offline_maps])
    pred_live = model.predict(X_live)
    pred_offline = model.predict(X_offline)
    score_live = -model.score_samples(X_live)
    score_offline = -model.score_samples(X_offline)

    flagged_live = pred_live == -1
    flagged_offline = pred_offline == -1
    flipped = flagged_live != flagged_offline
    n = len(table)

    print("\n=== Anomaly classification: live-shuffled vs offline features ===")
    print(
        f"rows: {n} | flagged offline={int(flagged_offline.sum())} ({flagged_offline.mean():.2%}) | "
        f"flagged live={int(flagged_live.sum())} ({flagged_live.mean():.2%})"
    )
    print(f"rows whose verdict FLIPPED: {int(flipped.sum())}/{n} ({flipped.mean():.2%})")

    example_indices = np.flatnonzero(flipped)[:8]
    for idx in example_indices:
        row = table.iloc[idx]
        deltas = {
            feature: float(row[f"{feature}__delta"])
            for feature in FEATURES
            if not np.isnan(row[f"{feature}__delta"])
        }
        deltas_str = ", ".join(f"{k}={v:+.3f}" for k, v in deltas.items())
        print(
            f"  order {row[ORDER_ITEM_ID]}: offline={'ANOM' if pred_offline[idx] == -1 else 'normal'} "
            f"(score {score_offline[idx]:+.3f}) -> live={'ANOM' if pred_live[idx] == -1 else 'normal'} "
            f"(score {score_live[idx]:+.3f}) | deltas: {deltas_str}"
        )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-size", type=int, default=150, help="number of replayed stream orders")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="replay in the producer's shuffled order (random_state=42) instead of chronological",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    raw = pd.read_csv(RAW_DATA_PATH, encoding="latin1")
    raw[ORDER_ITEM_ID] = raw[ORDER_ITEM_ID].astype(str)
    raw[ORDER_DATE_COLUMN] = pd.to_datetime(raw[ORDER_DATE_COLUMN], errors="coerce")
    raw = raw.dropna(subset=[ORDER_DATE_COLUMN]).reset_index(drop=True)

    if args.shuffle:

        stream = (
            raw.sample(frac=1.0, random_state=args.shuffle_seed)
            .reset_index(drop=True)
            .head(args.stream_size)
        )
        seed_df = raw  
        stream_ids: set[str] = set()
        order_note = f"shuffled (producer order, random_state={args.shuffle_seed})"
    else:
        sorted_raw = raw.sort_values(ORDER_DATE_COLUMN, kind="mergesort").reset_index(drop=True)
        stream_ids = set(sorted_raw[ORDER_ITEM_ID].tail(args.stream_size))
        stream = sorted_raw[sorted_raw[ORDER_ITEM_ID].isin(stream_ids)].reset_index(drop=True)
        seed_df = sorted_raw
        order_note = "chronological"

    ref = _build_offline_reference(raw)

    live_state = LiveFeatureState()
    live_state.seed_from_historical(
        seed_df, exclude_order_item_ids=stream_ids if stream_ids else None
    )
    print(
        f"[verify] seeded live state: price_groups={len(live_state.price_state)} "
        f"categories={len(live_state.delivery_stats)} "
        f"inventory_pairs={len(live_state.inventory_state)}"
    )
    print(f"[verify] replaying {len(stream)} stream orders ({order_note})")

    table = _replay(live_state, stream, ref)

    n_pairs = table.groupby(["product", "region"]).ngroups
    n_groups = table.groupby(["region", "category"]).ngroups
    print(
        f"[verify] stream covers {table['region'].nunique()} regions / "
        f"{table['category'].nunique()} categories / {n_groups} (region, category) "
        f"groups / {n_pairs} (product, region) pairs"
    )

    print("\n=== Verification summary ===")
    all_exact = True
    for feature in FEATURES:
        matches = table[f"{feature}__match"]
        exact = int(matches.sum())
        total = len(table)
        deltas = table.loc[matches, f"{feature}__delta"].abs()
        max_delta = float(deltas.max()) if len(deltas) else 0.0
        ok = exact == total
        all_exact = all_exact and ok
        print(
            f"{feature:<45} exact={exact}/{total}  max_abs_delta={max_delta:.3e}  {'PASS' if ok else 'FAIL'}"
        )

    if args.shuffle:
        print("\n=== Divergence under shuffled (out-of-order) arrival ===")
        for feature in FEATURES:
            matches = table[f"{feature}__match"]
            exact = int(matches.sum())
            total = len(table)
            diverged = table.loc[~matches, f"{feature}__delta"].abs()
            if len(diverged):
                print(
                    f"{feature:<45} exact={exact}/{total} ({exact / total:.1%}) | "
                    f"median_delta={diverged.median():.3e} mean_delta={diverged.mean():.3e} "
                    f"max_delta={diverged.max():.3e}"
                )
            else:
                print(f"{feature:<45} exact={exact}/{total} ({exact / total:.1%}) | no divergence")
        _flip_analysis(table)

    print("\n=== Side-by-side comparison (first 12 stream orders) ===")
    show = table.head(12).copy()
    for feature in FEATURES:
        show[feature] = show.apply(
            lambda r: (
                f"{r[f'{feature}__live']:.4f} | {r[f'{feature}__offline']:.4f} | {r[f'{feature}__delta']:.2e}"
                if np.isfinite(r[f"{feature}__delta"])
                else f"{r[f'{feature}__live']:.4f} | {r[f'{feature}__offline']:.4f} |    n/a"
            ),
            axis=1,
        )
    columns = [ORDER_ITEM_ID, "order_date", "region", "category", "product"] + FEATURES
    print(show[columns].to_string(index=False, max_colwidth=40))

    if not args.shuffle and TRAINED_CSV_PATH.exists():
        trained = pd.read_csv(TRAINED_CSV_PATH, encoding="latin1")
        trained[ORDER_ITEM_ID] = trained[ORDER_ITEM_ID].astype(str)
        trained_lookup = trained.set_index(ORDER_ITEM_ID)["stock_after_order"]
        present = sum(1 for oid in table[ORDER_ITEM_ID] if oid in trained_lookup.index)
        matched = sum(
            1
            for oid, live_val in zip(table[ORDER_ITEM_ID], table["stock_after_order__live"])
            if oid in trained_lookup.index and _close(live_val, float(trained_lookup.loc[oid]))
        )
        print(
            f"\n=== Cross-check vs training CSV ===\n"
            f"stream orders present in dataset_with_anomalies.csv: {present}/{len(table)}\n"
            f"live stock_after_order identical to training CSV: {matched}/{present} "
            f"(the rest are pairs whose base stock changed because synthetic "
            f"inventory-depletion burst orders were injected into their restock windows)"
        )

    rows_exact = table[[f"{f}__match" for f in FEATURES]].all(axis=1)
    print()
    if all_exact:
        print(
            f"RESULT: PASS - all {len(table)} stream orders match the offline pipeline "
            f"on all {len(FEATURES)} features"
        )
        sys.exit(0)
    print(
        f"RESULT: FAIL - {int((~rows_exact).sum())} of {len(table)} stream orders differ "
        f"from the offline pipeline"
    )
    sys.exit(1)

if __name__ == "__main__":
    main()
