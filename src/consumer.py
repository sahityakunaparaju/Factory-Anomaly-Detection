from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer

from src.db import insert_anomaly
from src.live_state import FEATURE_COLUMNS, LiveFeatureState
from src.model_registry import load_manifest, resolve_model_path

TOPIC_NAME = os.getenv("KAFKA_TOPIC", "factory-orders")
BOOTSTRAP_SERVERS = [os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")]

SCALER_PATH = Path(os.getenv("SCALER_PATH", "models/lof_mad_price_scaler.joblib"))
MODEL_MANIFEST_PATH = Path(os.getenv("MODEL_MANIFEST_PATH", "models/current_model.json"))

MODEL_PATH = os.getenv("MODEL_PATH", "")
DB_PATH = Path(os.getenv("ANOMALY_DB_PATH", "data/anomalies.db"))
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/DataCoSupplyChainDataset.csv"))
FEATURE_LOG_PATH = Path(os.getenv("FEATURE_LOG_PATH", "data/consumer_features.jsonl"))

CONSUMER_STATE_PATH = Path(os.getenv("CONSUMER_STATE_PATH", "data/consumer_state.json"))
CONSUMER_TIMEOUT_MS = int(os.getenv("CONSUMER_TIMEOUT_MS", "3000"))

EXCLUDE_ORDER_IDS_PATH = os.getenv("EXCLUDE_ORDER_IDS_PATH") or ""

_MODEL_MANIFEST = load_manifest(MODEL_MANIFEST_PATH)

def _resolve_model_path() -> Path:
    """Resolve the model file: explicit MODEL_PATH > manifest > default."""
    if MODEL_PATH:
        return Path(MODEL_PATH)
    return resolve_model_path(MODEL_MANIFEST_PATH)

_manifest_threshold: float | None = None
if _MODEL_MANIFEST:
    try:
        _manifest_threshold = float(_MODEL_MANIFEST["threshold"])
    except (KeyError, TypeError, ValueError):
        _manifest_threshold = None
ANOMALY_THRESHOLD = (os.getenv("ANOMALY_THRESHOLD") or "").strip()
ANOMALY_THRESHOLD = float(ANOMALY_THRESHOLD) if ANOMALY_THRESHOLD else _manifest_threshold

SOURCE_TAG = "env" if (os.getenv("ANOMALY_THRESHOLD") or "").strip() else "manifest"

def _write_consumer_state(
    orders_processed: int = 0,
    started_at: str | None = None,
    last_processed_at: str | None = None,
) -> None:
    """Record the threshold the consumer is really flagging with.

    ``ANOMALY_THRESHOLD`` env wins; otherwise the manifest's native threshold
    (``-offset_``) is the fallback; if neither exists the model's
    ``predict()`` labels are used directly. The manifest threshold alone is
    ambiguous (it is a fallback, not the operative value when the env var is
    set), so the dashboard reads this file as the single source of truth.
    """
    if ANOMALY_THRESHOLD is not None:
        threshold, source = float(ANOMALY_THRESHOLD), SOURCE_TAG
    else:
        threshold, source = None, "model-predict"
    payload = {
        "model_file": _resolve_model_path().name,
        "threshold": threshold,
        "threshold_source": source,
        "started_at": started_at,
        "last_processed_at": last_processed_at,
        "orders_processed": orders_processed,
    }
    try:
        CONSUMER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONSUMER_STATE_PATH.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass

def _prepare_features(feature_map: dict[str, object], scaler_bundle: dict[str, object]) -> np.ndarray:
    """Scale the live-computed features exactly like training did (MAD price + standard rest)."""
    row_values = np.array(
        [[feature_map[column] for column in FEATURE_COLUMNS]],
        dtype=float,
    )
    row_values = np.nan_to_num(row_values, nan=0.0, posinf=0.0, neginf=0.0)

    price_index = int(scaler_bundle["price_index"])
    price_median = float(scaler_bundle["price_median"])

    price_scale = scaler_bundle.get("price_scale")
    if price_scale is None:
        price_scale = 1.4826 * float(scaler_bundle["price_mad"])
    price_scale = float(price_scale)
    if price_scale == 0 or not np.isfinite(price_scale):
        price_scale = 1.0

    row_scaled = row_values.copy()
    row_scaled[:, price_index] = (row_values[:, price_index] - price_median) / price_scale

    non_price_columns = [idx for idx in range(len(FEATURE_COLUMNS)) if idx != price_index]
    if non_price_columns:
        row_scaled[:, non_price_columns] = scaler_bundle["non_price_scaler"].transform(
            row_values[:, non_price_columns]
        )

    return row_scaled

def _build_reason(row_scaled: np.ndarray) -> str:
    scores = np.abs(row_scaled[0])
    ranked = sorted(zip(FEATURE_COLUMNS, scores), key=lambda item: item[1], reverse=True)
    top = ranked[:3]
    parts = [f"{feature} deviated strongly ({score:.3f})" for feature, score in top]
    return "; ".join(parts)

def _log_features(order_id: str | int, row_index: int | None, feature_map: dict[str, object]) -> None:
    """Append the live-computed features to a JSONL file for post-hoc verification."""
    FEATURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": str(order_id),
        "row_index": row_index,
        "features": feature_map,
    }
    with FEATURE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")

def main() -> None:
    model_path = _resolve_model_path()
    model = joblib.load(model_path)
    scaler_bundle = joblib.load(SCALER_PATH)
    if _MODEL_MANIFEST:
        print(
            f"[consumer] model from manifest: {_MODEL_MANIFEST.get('model_file')} "
            f"(threshold fallback {_MODEL_MANIFEST.get('threshold')})"
        )
    else:
        print(f"[consumer] loaded model: {model_path}")
    if ANOMALY_THRESHOLD is not None:
        print(f"[consumer] flagging threshold: {ANOMALY_THRESHOLD:.4f} ({SOURCE_TAG})")

    started_at = datetime.now(timezone.utc).isoformat()
    processed_orders = 0
    _write_consumer_state(started_at=started_at)

    seed_df = pd.read_csv(RAW_DATA_PATH, encoding="latin1")
    seed_kwargs: dict[str, object] = {}
    if EXCLUDE_ORDER_IDS_PATH:
        exclude_ids = pd.read_csv(EXCLUDE_ORDER_IDS_PATH, header=None, dtype=str)[0]
        exclude_ids = set(exclude_ids.astype(str))
        seed_kwargs["exclude_order_item_ids"] = exclude_ids
        print(
            f"[consumer] excluding {len(exclude_ids)} stream order ids from the "
            f"causal seed ({EXCLUDE_ORDER_IDS_PATH})"
        )
    live_state = LiveFeatureState()
    live_state.seed_from_historical(seed_df, **seed_kwargs)
    print(f"[consumer] seeded live feature state from historical dataset: {RAW_DATA_PATH}")
    print(
        f"[consumer] state size: price_groups={len(live_state.price_state)} "
        f"categories={len(live_state.delivery_stats)} "
        f"inventory_pairs={len(live_state.inventory_state)}"
    )

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS,
    )

    print(f"[consumer] connected to topic={TOPIC_NAME}")

    try:
        for message in consumer:
            processed_orders += 1
            payload = message.value
            order_id = payload.get("order_id")
            row_index = payload.get("row_index")
            row = payload.get("payload", {})

            feature_map = live_state.compute_features(row)
            _log_features(order_id, row_index, feature_map)

            row_scaled = _prepare_features(feature_map, scaler_bundle)
            pred_label = int(model.predict(row_scaled)[0])
            anomaly_score = float(-model.score_samples(row_scaled)[0])
            if ANOMALY_THRESHOLD is not None:
                flagged = anomaly_score >= ANOMALY_THRESHOLD
            else:
                flagged = pred_label == -1
            reason = _build_reason(row_scaled)

            print(
                f"[consumer] processed order_id={order_id} flagged={flagged} "
                f"score={anomaly_score:.6f} reason={reason}"
            )

            if flagged:
                insert_anomaly(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    order_id=order_id,
                    anomaly_score=anomaly_score,
                    reason=reason,
                    db_path=DB_PATH,
                )
    finally:
        _write_consumer_state(
            orders_processed=processed_orders,
            started_at=started_at,
            last_processed_at=datetime.now(timezone.utc).isoformat(),
        )
        consumer.close()

if __name__ == "__main__":
    main()
