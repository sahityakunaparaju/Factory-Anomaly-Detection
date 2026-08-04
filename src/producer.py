from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaProducer

DATA_PATH = Path(os.getenv("PRODUCER_DATA_PATH", "data/processed/dataset_with_anomalies.csv"))
TOPIC_NAME = "factory-orders"
BOOTSTRAP_SERVERS = ["localhost:9092"]
DELAY_SECONDS = float(os.getenv("PRODUCER_DELAY_SECONDS", "1.5"))

MAX_MESSAGES = int(os.getenv("PRODUCER_MAX_MESSAGES", "0")) or None

SHUFFLE_WINDOW_DAYS = int(os.getenv("PRODUCER_SHUFFLE_WINDOW_DAYS", "0"))
ORDER_DATE_COLUMN = "order date (DateOrders)"
ORDER_ITEM_ID_COLUMN = "Order Item Id"
RANDOM_STATE = 42

ORDER_IDS_PATH = Path(os.getenv("PRODUCER_ORDER_IDS_PATH", "data/stream_order_ids.txt"))

RAW_FIELD_COLUMNS = [
    "Order Id",
    "Order Item Id",
    "Product Card Id",
    "Order Region",
    "Category Name",
    "Supplier Name",
    "Order Item Product Price",
    "Order Item Quantity",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Item Discount Rate",
    "order date (DateOrders)",
]

def build_stream(
    data_path: str | Path = DATA_PATH,
    shuffle_window_days: int = SHUFFLE_WINDOW_DAYS,
    random_state: int = RANDOM_STATE,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load the (injected) dataset and order it for streaming.

    Rows are returned in chronological order by default. When
    ``shuffle_window_days > 0``, rows are shuffled only inside each rolling
    N-day block (block order stays chronological), so arrival order remains
    near-chronological. ``max_rows`` truncates to the chronologically-earliest
    prefix, which is how PRODUCER_MAX_MESSAGES keeps smoke runs short.
    """
    df = pd.read_csv(data_path, encoding="latin1")

    df[ORDER_DATE_COLUMN] = pd.to_datetime(df[ORDER_DATE_COLUMN], errors="coerce", format="mixed")
    df = df.dropna(subset=[ORDER_DATE_COLUMN]).reset_index(drop=True)
    df = df.sort_values(ORDER_DATE_COLUMN, kind="mergesort").reset_index(drop=True)

    if shuffle_window_days and shuffle_window_days > 0:
        days = (df[ORDER_DATE_COLUMN].dt.normalize() - df[ORDER_DATE_COLUMN].dt.normalize().min()).dt.days
        block = (days // shuffle_window_days).astype(int)
        blocks = [
            block_df.sample(frac=1.0, random_state=random_state, ignore_index=True)
            for _, block_df in df.groupby(block, sort=True)
        ]
        df = pd.concat(blocks, ignore_index=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).reset_index(drop=True)
    return df

def main() -> None:
    df = build_stream(data_path=DATA_PATH, shuffle_window_days=SHUFFLE_WINDOW_DAYS, max_rows=MAX_MESSAGES)
    print(
        f"[producer] streaming {len(df)} orders from {DATA_PATH} "
        f"(shuffle_window_days={SHUFFLE_WINDOW_DAYS})"
    )

    ORDER_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORDER_IDS_PATH.write_text(
        "\n".join(df[ORDER_ITEM_ID_COLUMN].astype(str)) + "\n", encoding="utf-8"
    )
    print(f"[producer] wrote {len(df)} stream order ids to {ORDER_IDS_PATH}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
    )

    try:
        for row_index, (_, row) in enumerate(df.iterrows()):
            payload = {column: row.get(column) for column in RAW_FIELD_COLUMNS if column in row}
            message = {
                "row_index": int(row_index),
                "order_id": row.get("Order Id"),
                "payload": payload,
            }
            producer.send(TOPIC_NAME, value=message)
            print(
                f"[producer] sent row_index={row_index} order_id={message['order_id']} "
                f"raw_fields={list(payload.keys())}"
            )
            producer.flush()
            time.sleep(DELAY_SECONDS)
    finally:
        producer.close()

if __name__ == "__main__":
    main()
