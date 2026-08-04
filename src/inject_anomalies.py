"""Synthetic anomaly injection utilities for the factory anomaly detection workflow.

This module keeps anomaly generation separate from feature generation. It builds
an independent synthetic inventory proxy, then injects three anomaly families on
non-overlapping random row subsets and recomputes the engineered features on the
modified dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_feature_dataframe

PRODUCT_ID_COLUMN = "Product Card Id"
REGION_COLUMN = "Order Region"
QUANTITY_COLUMN = "Order Item Quantity"
PRICE_COLUMN = "Order Item Product Price"
REAL_SHIPPING_COLUMN = "Days for shipping (real)"
ORDER_DATE_COLUMN = "order date (DateOrders)"

def inventory_pair_constants(
    df: pd.DataFrame,
    restock_interval_days: int = 14,
    base_stock_multiple: int = 2,
    quantity_column: str = QUANTITY_COLUMN,
    product_column: str = PRODUCT_ID_COLUMN,
    region_column: str = REGION_COLUMN,
    order_date_column: str = ORDER_DATE_COLUMN,
) -> pd.DataFrame:
    """Compute the frozen per-(product, region) inventory constants.

    Returns a frame indexed by the pair key (``"<product> | <region>"``) with
    the base stock level and the first order date for every product/region
    pair. Both :func:`simulate_inventory` (offline) and the streaming
    consumer's live state seed from these same constants so the live
    simulation matches the offline one exactly.
    """
    required_columns = [product_column, region_column, quantity_column, order_date_column]
    if not all(column in df.columns for column in required_columns):
        missing = [column for column in required_columns if column not in df.columns]
        raise ValueError(f"inventory_pair_constants() requires columns: {missing}")

    ordered = df.copy()
    ordered[order_date_column] = pd.to_datetime(ordered[order_date_column], errors="coerce")
    ordered = ordered.dropna(subset=[product_column, region_column, order_date_column]).copy()
    ordered = ordered.sort_values(
        by=[product_column, region_column, order_date_column],
        kind="mergesort",
    ).reset_index(drop=True)

    pair_key = ordered[product_column].astype(str).str.cat(
        ordered[region_column].astype(str), sep=" | "
    )
    ordered["__pair_key"] = pair_key

    pair_first_date = ordered.groupby("__pair_key")[order_date_column].transform("min")
    ordered["__restock_window_id"] = (
        (ordered[order_date_column] - pair_first_date).dt.days // restock_interval_days
    )

    pair_avg_qty = ordered.groupby("__pair_key")[quantity_column].mean()
    pair_orders_per_window = (
        ordered.groupby(["__pair_key", "__restock_window_id"]).size().groupby(level=0).mean()
    )
    base_stock_pair = np.maximum(
        1.0,
        np.round(pair_orders_per_window.mul(pair_avg_qty).mul(base_stock_multiple)),
    ).astype(float)
    first_date_pair = ordered.groupby("__pair_key")[order_date_column].min()

    constants = pd.DataFrame(
        {"base_stock": base_stock_pair, "first_order_date": first_date_pair}
    )
    key_map = (
        ordered[["__pair_key", product_column, region_column]]
        .drop_duplicates("__pair_key")
        .set_index("__pair_key")
    )
    return constants.join(key_map)

def simulate_inventory(
    df: pd.DataFrame,
    restock_interval_days: int = 14,
    base_stock_multiple: int = 2,
    quantity_column: str = QUANTITY_COLUMN,
    product_column: str = PRODUCT_ID_COLUMN,
    region_column: str = REGION_COLUMN,
    order_date_column: str = ORDER_DATE_COLUMN,
) -> pd.DataFrame:
    """Simulate a synthetic stock-level proxy for each (product, region) series.

    The series is generated from fixed calendar windows rather than inter-order
    gaps, so the inventory proxy is realistic and does not depend on anomaly
    selection.
    """
    required_columns = [product_column, region_column, quantity_column, order_date_column]
    if not all(column in df.columns for column in required_columns):
        missing = [column for column in required_columns if column not in df.columns]
        raise ValueError(f"simulate_inventory() requires columns: {missing}")

    constants = inventory_pair_constants(
        df,
        restock_interval_days=restock_interval_days,
        base_stock_multiple=base_stock_multiple,
        quantity_column=quantity_column,
        product_column=product_column,
        region_column=region_column,
        order_date_column=order_date_column,
    )

    ordered = df.copy()
    ordered[order_date_column] = pd.to_datetime(ordered[order_date_column], errors="coerce")
    ordered = ordered.dropna(subset=[product_column, region_column, order_date_column]).copy()
    ordered = ordered.sort_values(
        by=[product_column, region_column, order_date_column],
        kind="mergesort",
    ).reset_index(drop=True)

    pair_key = ordered[product_column].astype(str).str.cat(
        ordered[region_column].astype(str), sep=" | "
    )
    ordered["__pair_key"] = pair_key

    ordered["base_stock"] = ordered["__pair_key"].map(constants["base_stock"])
    ordered["__restock_window_id"] = (
        (ordered[order_date_column] - ordered["__pair_key"].map(constants["first_order_date"])).dt.days
        // restock_interval_days
    )

    ordered["__cum_qty_in_window"] = ordered.groupby(["__pair_key", "__restock_window_id"])[quantity_column].cumsum()
    ordered["stock_after_order"] = ordered["base_stock"] - ordered["__cum_qty_in_window"]

    recent_demand = (
        ordered.groupby("__pair_key")[quantity_column]
        .transform(lambda series: series.shift(1).rolling(window=7, min_periods=1).mean())
    )
    recent_demand = recent_demand.fillna(ordered[quantity_column])

    ordered["days_of_cover_remaining"] = np.where(
        recent_demand > 0,
        ordered["stock_after_order"] / recent_demand,
        np.nan,
    )

    ordered["stock_decline_rate"] = (
        -ordered.groupby("__pair_key")["stock_after_order"].diff().fillna(0.0)
    )

    ordered.drop(
        columns=["__pair_key", "__restock_window_id", "__cum_qty_in_window", "base_stock"],
        inplace=True,
    )
    return ordered

def _select_non_overlapping_row_subsets(
    n_rows: int,
    target_fraction: float,
    rng: np.random.Generator,
    subset_count: int = 3,
) -> list[np.ndarray]:
    """Return non-overlapping random row index arrays with a fixed size each."""
    target_size = max(1, int(round(target_fraction * n_rows)))
    remaining = np.arange(n_rows)
    subsets: list[np.ndarray] = []

    for _ in range(subset_count):
        subset = rng.choice(remaining, size=target_size, replace=False)
        subsets.append(subset)
        remaining = np.setdiff1d(remaining, subset, assume_unique=True)

    return subsets

def inject_anomalies(
    df: pd.DataFrame,
    random_state: int = 42,
    anomaly_fraction: float = 0.015,
    price_multiplier_low: float = 1.25,
    price_multiplier_high: float = 1.50,
    delay_low_days: int = 10,
    delay_high_days: int = 15,
    restock_interval_days: int = 14,
) -> pd.DataFrame:
    """Inject three non-overlapping anomaly families into the dataset."""
    rng = np.random.default_rng(random_state)

    frozen_feature_frame = build_feature_dataframe(df)
    frozen_baseline_lookup = (
        frozen_feature_frame[
            ["Order Item Id", PRICE_COLUMN, "price_deviation_from_supplier_category_avg"]
        ]
        .drop_duplicates(subset=["Order Item Id"])
        .copy()
    )

    frozen_baseline_lookup["frozen_baseline"] = (
        pd.to_numeric(frozen_baseline_lookup[PRICE_COLUMN], errors="coerce")
        - frozen_baseline_lookup["price_deviation_from_supplier_category_avg"]
    )
    modified = df.copy()

    modified["is_anomaly"] = 0
    modified["anomaly_type"] = "normal"

    n_rows = len(modified)
    subset_indices = _select_non_overlapping_row_subsets(
        n_rows=n_rows,
        target_fraction=anomaly_fraction,
        rng=rng,
        subset_count=3,
    )

    price_indices = subset_indices[0]
    delivery_indices = subset_indices[1]
    inventory_indices = subset_indices[2]

    modified.loc[price_indices, PRICE_COLUMN] = modified.loc[price_indices, PRICE_COLUMN].mul(
        rng.uniform(price_multiplier_low, price_multiplier_high, size=len(price_indices))
    )
    modified.loc[price_indices, "is_anomaly"] = 1
    modified.loc[price_indices, "anomaly_type"] = "price_spike"

    delay_extra = rng.integers(delay_low_days, delay_high_days + 1, size=len(delivery_indices))
    modified.loc[delivery_indices, REAL_SHIPPING_COLUMN] = (
        pd.to_numeric(modified.loc[delivery_indices, REAL_SHIPPING_COLUMN], errors="coerce") + delay_extra
    )
    modified.loc[delivery_indices, "is_anomaly"] = 1
    modified.loc[delivery_indices, "anomaly_type"] = "delivery_delay"

    inventory_frame = modified.copy()
    inventory_frame[ORDER_DATE_COLUMN] = pd.to_datetime(inventory_frame[ORDER_DATE_COLUMN], errors="coerce")
    inventory_frame = inventory_frame.dropna(subset=[PRODUCT_ID_COLUMN, REGION_COLUMN, ORDER_DATE_COLUMN]).copy()
    inventory_frame = inventory_frame.sort_values(
        by=[PRODUCT_ID_COLUMN, REGION_COLUMN, ORDER_DATE_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)

    pair_key = inventory_frame[PRODUCT_ID_COLUMN].astype(str).str.cat(
        inventory_frame[REGION_COLUMN].astype(str), sep=" | "
    )
    inventory_frame["__pair_key"] = pair_key
    pair_first_date = inventory_frame.groupby("__pair_key")[ORDER_DATE_COLUMN].transform("min")
    inventory_frame["__restock_window_id"] = (
        (inventory_frame[ORDER_DATE_COLUMN] - pair_first_date).dt.days // restock_interval_days
    )

    inventory_seed_rows = inventory_frame.loc[inventory_indices].copy()
    inventory_window_keys = (
        inventory_seed_rows[["__pair_key", "__restock_window_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    burst_rows: list[pd.DataFrame] = []
    next_order_id = int(pd.to_numeric(modified["Order Id"], errors="coerce").max()) + 1
    next_order_item_id = int(pd.to_numeric(modified["Order Item Id"], errors="coerce").max()) + 1

    for _, window_row in inventory_window_keys.iterrows():
        pair = window_row["__pair_key"]
        window_id = int(window_row["__restock_window_id"])
        window_slice = inventory_frame[
            (inventory_frame["__pair_key"] == pair)
            & (inventory_frame["__restock_window_id"] == window_id)
        ].copy()

        if window_slice.empty:
            continue

        burst_size = int(rng.integers(3, 6))
        median_qty = float(window_slice[QUANTITY_COLUMN].median())
        if pd.isna(median_qty) or median_qty <= 0:
            median_qty = float(window_slice[QUANTITY_COLUMN].mean())

        pair_qty = pd.to_numeric(window_slice[QUANTITY_COLUMN], errors="coerce").dropna()
        qty_center = int(round(float(median_qty)))
        qty_floor = max(1, int(np.floor(pair_qty.min())) - 1) if not pair_qty.empty else max(1, qty_center - 1)
        qty_ceiling = max(qty_floor + 1, int(np.ceil(pair_qty.max())) + 1) if not pair_qty.empty else qty_center + 1
        qty_floor = max(1, min(qty_floor, qty_center))
        qty_ceiling = max(qty_center + 1, qty_ceiling)
        if qty_floor == qty_ceiling:
            qty_floor = max(1, qty_floor - 1)
            qty_ceiling = qty_ceiling + 1

        price_values = pd.to_numeric(window_slice[PRICE_COLUMN], errors="coerce").dropna()
        price_center = float(price_values.median()) if not price_values.empty else float(window_slice[PRICE_COLUMN].median())
        price_base_low = float(price_values.min()) if not price_values.empty else max(price_center * 0.95, 0.0)
        price_base_high = float(price_values.max()) if not price_values.empty else price_center * 1.05
        price_jitter_scale = max(abs(price_base_high - price_base_low) * 0.05, max(price_center * 0.02, 0.05))
        price_low = max(price_base_low - price_jitter_scale, 0.0)
        price_high = price_base_high + price_jitter_scale
        baseline_price = float(window_slice[PRICE_COLUMN].median())

        window_start = window_slice[ORDER_DATE_COLUMN].min()
        window_end = window_slice[ORDER_DATE_COLUMN].max()
        if pd.isna(window_start) or pd.isna(window_end):
            continue

        burst_dates = pd.to_datetime(
            pd.date_range(start=window_start, end=window_end, periods=burst_size + 2)[1:-1]
        )
        burst_frame = window_slice.iloc[[0]].copy()
        burst_frame = pd.concat([burst_frame] * burst_size, ignore_index=True)

        burst_frame[ORDER_DATE_COLUMN] = burst_dates.strftime("%m/%d/%Y %H:%M")

        burst_frame[QUANTITY_COLUMN] = rng.integers(qty_floor, qty_ceiling + 1, size=burst_size).astype(float)
        burst_frame[PRICE_COLUMN] = np.clip(
            price_center + rng.uniform(-price_jitter_scale, price_jitter_scale, size=burst_size),
            price_low,
            price_high,
        )
        burst_frame["price_deviation_from_supplier_category_avg"] = (
            burst_frame[PRICE_COLUMN].astype(float) - baseline_price
        )

        if "Order Item Discount" in burst_frame.columns:
            discount_values = pd.to_numeric(burst_frame["Order Item Discount"], errors="coerce").fillna(0.0)
            discount_low = float(window_slice["Order Item Discount"].min()) if "Order Item Discount" in window_slice.columns else 0.0
            discount_high = float(window_slice["Order Item Discount"].max()) if "Order Item Discount" in window_slice.columns else 0.0
            discount_span = max((discount_high - discount_low) * 0.5, 0.5)
            burst_frame["Order Item Discount"] = np.clip(
                discount_values + rng.uniform(-discount_span, discount_span, size=burst_size),
                max(0.0, discount_low),
                max(0.0, discount_high),
            )

        if "Order Item Discount Rate" in burst_frame.columns:
            discount_rate_values = pd.to_numeric(burst_frame["Order Item Discount Rate"], errors="coerce").fillna(0.0)
            discount_rate_low = float(window_slice["Order Item Discount Rate"].min()) if "Order Item Discount Rate" in window_slice.columns else 0.0
            discount_rate_high = float(window_slice["Order Item Discount Rate"].max()) if "Order Item Discount Rate" in window_slice.columns else 0.0
            discount_rate_span = max((discount_rate_high - discount_rate_low) * 0.5, 0.01)
            burst_frame["Order Item Discount Rate"] = np.clip(
                discount_rate_values + rng.uniform(-discount_rate_span, discount_rate_span, size=burst_size),
                max(0.0, discount_rate_low),
                max(0.0, discount_rate_high),
            )

        burst_frame["Order Id"] = np.arange(next_order_id, next_order_id + burst_size)
        burst_frame["Order Item Id"] = np.arange(next_order_item_id, next_order_item_id + burst_size)
        next_order_id += burst_size
        next_order_item_id += burst_size

        burst_frame["is_anomaly"] = 1
        burst_frame["anomaly_type"] = "inventory_depletion"

        burst_frame.drop(columns=["__pair_key", "__restock_window_id"], errors="ignore", inplace=True)
        burst_rows.append(burst_frame)

    if burst_rows:
        injected_inventory = pd.concat(burst_rows, ignore_index=True)
        modified = pd.concat([modified, injected_inventory], ignore_index=True)
        modified["is_anomaly"] = modified["is_anomaly"].fillna(0)
        modified["anomaly_type"] = modified["anomaly_type"].fillna("normal")

    simulated = simulate_inventory(modified)

    feature_frame = build_feature_dataframe(modified)
    feature_frame = feature_frame.merge(
        frozen_baseline_lookup[["Order Item Id", "frozen_baseline"]],
        on="Order Item Id",
        how="left",
    )
    feature_frame["price_deviation_from_supplier_category_avg"] = (
        pd.to_numeric(feature_frame[PRICE_COLUMN], errors="coerce")
        - feature_frame["frozen_baseline"]
    ).fillna(feature_frame["price_deviation_from_supplier_category_avg"])
    feature_frame.drop(columns=["frozen_baseline"], inplace=True)

    if burst_rows:
        synthetic_price_lookup = pd.concat(
            [burst_frame[["Order Item Id", "price_deviation_from_supplier_category_avg"]] for burst_frame in burst_rows],
            ignore_index=True,
        )
        feature_frame = feature_frame.merge(
            synthetic_price_lookup,
            on="Order Item Id",
            how="left",
            suffixes=("", "_synthetic"),
        )
        feature_frame["price_deviation_from_supplier_category_avg"] = (
            feature_frame["price_deviation_from_supplier_category_avg_synthetic"]
            .fillna(feature_frame["price_deviation_from_supplier_category_avg"])
        )
        feature_frame.drop(columns=["price_deviation_from_supplier_category_avg_synthetic"], inplace=True)

    simulated_join = simulated[
        ["Order Item Id", "stock_after_order", "days_of_cover_remaining", "stock_decline_rate"]
    ]
    feature_frame = feature_frame.merge(simulated_join, on="Order Item Id", how="left")

    feature_frame[["stock_after_order", "days_of_cover_remaining", "stock_decline_rate"]] = feature_frame[
        ["stock_after_order", "days_of_cover_remaining", "stock_decline_rate"]
    ].fillna(0.0)

    feature_frame["is_anomaly"] = modified["is_anomaly"].astype(int)
    feature_frame["anomaly_type"] = modified["anomaly_type"].astype(str)

    return feature_frame

def summarize_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize anomaly counts and percentages by type."""
    summary = (
        df.groupby("anomaly_type")
        .size()
        .rename("count")
        .reset_index()
    )
    summary["percent"] = (summary["count"] / len(df)) * 100
    return summary

if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv("data/raw/DataCoSupplyChainDataset.csv", encoding="latin1")
    anomaly_frame = inject_anomalies(data)
    elapsed = time.perf_counter() - start_time

    output_path = Path("data/processed/dataset_with_anomalies.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_frame.to_csv(output_path, index=False)

    summary = summarize_anomalies(anomaly_frame)
    print(f"Anomaly injection runtime on full dataset: {elapsed:.3f} seconds")
    print(summary.to_string(index=False))

    overlaps = anomaly_frame.groupby("anomaly_type")["is_anomaly"].sum()
    print("rows with more than one anomaly_type:", int((anomaly_frame["is_anomaly"] > 1).sum()))
    print("Saved dataset to:", output_path)
