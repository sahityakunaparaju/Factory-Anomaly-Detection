"""Feature engineering utilities for the factory anomaly detection pipeline.

This module contains the first pipeline step: creating order-level anomaly
signals from the raw supply-chain dataset. The functions are intentionally
modular so they can be extended later with anomaly injection, modeling, and
training workflows.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

PRICE_COLUMN = "Order Item Product Price"
CATEGORY_COLUMN = "Category Name"
QUANTITY_COLUMN = "Order Item Quantity"
SHIPPING_REAL_COLUMN = "Days for shipping (real)"
SHIPPING_SCHEDULED_COLUMN = "Days for shipment (scheduled)"
DISCOUNT_RATE_COLUMN = "Order Item Discount Rate"

def _safe_zscore(series: pd.Series) -> pd.Series:
    """Return a z-score series with a stable fallback when a column is constant."""
    mean = series.mean()
    std = series.std(ddof=0)

    if pd.isna(mean) or pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index, dtype=float)

    return (series - mean) / std

def _robust_zscore(series: pd.Series) -> pd.Series:
    """Return a MAD-based robust z-score with a safe fallback when the MAD is zero."""
    median = series.median()
    mad = (series - median).abs().median()
    scale = 1.4826 * mad

    if pd.isna(median) or pd.isna(scale) or scale == 0:
        return pd.Series(0.0, index=series.index, dtype=float)

    return (series - median) / scale

def _resolve_supplier_column(df: pd.DataFrame) -> str:
    """Pick the best available supplier-like grouping field.

    Prefer an explicit supplier identifier when one exists. If not, fall back
    to the geographic order location that is the next-best signal for supplier
    behavior in this dataset.
    """
    candidates = [
        "Supplier Name",
        "Supplier ID",
        "Supplier Country",
        "Supplier State",
        "Order Region",
        "Order Country",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Unable to find a supplier-like column in the dataset. "
        "Expected one of: Supplier Name, Supplier ID, Order Region, or Order Country."
    )

def _sort_dataframe_for_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return the frame sorted by the most reliable time column available.

    The sort is STABLE (``mergesort``) and preserves the input row identity
    (the original index is kept, not reset). This is load-bearing: per-group
    rolling computations return Series aligned to the ORIGINAL rows, so
    callers assign the results back to the input frame by index instead of by
    position. An earlier version sorted with the default (unstable) quicksort
    and reset the index; with tens of thousands of duplicate timestamps the
    unstable tie order differed from the stable order every other pipeline
    stage uses, silently scrambling the price deviations onto the wrong rows
    (the "60% exact zeros" price column that later produced the degenerate
    MAD and the 1e-6 scale bug).
    """
    time_candidates = [
        "Order Date",
        "Shipping Date",
        "Order Created Date",
        "Order Date (DateOrders)",
        "order date (DateOrders)",
        "shipping date (DateOrders)",
        "Date",
    ]

    ordered = df.copy()
    for column in time_candidates:
        if column in ordered.columns:
            try:
                ordered[column] = pd.to_datetime(ordered[column], errors="coerce")
                ordered = ordered.sort_values(by=column, kind="mergesort")
                return ordered
            except Exception:
                continue

    return ordered

def compute_price_baseline_reference(
    df: pd.DataFrame,
    rolling_window: int = 30,
    price_column: str = PRICE_COLUMN,
    category_column: str = CATEGORY_COLUMN,
) -> pd.Series:
    """Compute a frozen, robust rolling median baseline by supplier/category.

    The baseline is intentionally robust to heavy-tailed outliers, so it uses a
    rolling median rather than a rolling mean and is safe to reuse across later
    anomaly injection steps.
    """
    ordered = _sort_dataframe_for_history(df)
    supplier_column = _resolve_supplier_column(ordered)

    rolling_baseline = (
        ordered.groupby([supplier_column, category_column], dropna=False)[price_column]
        .transform(
            lambda series: series.shift(1).rolling(window=rolling_window, min_periods=1).median()
        )
    )

    rolling_baseline = rolling_baseline.fillna(ordered[price_column])
    return rolling_baseline.astype(float)

def compute_price_deviation_by_supplier_category(
    df: pd.DataFrame,
    rolling_window: int = 30,
    price_column: str = PRICE_COLUMN,
    category_column: str = CATEGORY_COLUMN,
    baseline_reference: pd.Series | None = None,
) -> pd.Series:
    """Compute price deviation versus a frozen, robust supplier/category baseline.

    When a precomputed baseline reference is supplied, the current row's price is
    compared against that frozen value rather than re-rolling on the modified
    dataset after injection. This prevents synthetic anomalies from leaking into
    the baseline for nearby rows.
    """
    ordered = _sort_dataframe_for_history(df)
    supplier_column = _resolve_supplier_column(ordered)

    if baseline_reference is None:
        baseline_reference = compute_price_baseline_reference(
            ordered,
            rolling_window=rolling_window,
            price_column=price_column,
            category_column=category_column,
        )

    baseline_reference = pd.Series(baseline_reference, index=ordered.index, dtype=float)
    return ordered[price_column] - baseline_reference

def compute_order_quantity_zscore_by_category(
    df: pd.DataFrame,
    quantity_column: str = QUANTITY_COLUMN,
    category_column: str = CATEGORY_COLUMN,
) -> pd.Series:
    """Compute the z-score of order quantity relative to the historical
    quantity distribution for the same category.
    """
    grouped = df.groupby(category_column)[quantity_column]
    return grouped.transform(_safe_zscore)

def compute_delivery_delay_features(
    df: pd.DataFrame,
    shipping_real_column: str = SHIPPING_REAL_COLUMN,
    shipping_scheduled_column: str = SHIPPING_SCHEDULED_COLUMN,
    category_column: str = CATEGORY_COLUMN,
) -> pd.DataFrame:
    """Compute the delivery delay and its category-wise historical deviation."""
    feature_frame = df.copy()
    feature_frame["delivery_delay_days"] = (
        feature_frame[shipping_real_column] - feature_frame[shipping_scheduled_column]
    )

    feature_frame["delivery_delay_deviation"] = (
        feature_frame.groupby(category_column)["delivery_delay_days"].transform(_safe_zscore)
    )

    return feature_frame

def compute_discount_rate_anomaly(
    df: pd.DataFrame,
    discount_rate_column: str = DISCOUNT_RATE_COLUMN,
    category_column: str = CATEGORY_COLUMN,
) -> pd.Series:
    """Compute a category-relative anomaly score for discount rate."""
    return df.groupby(category_column)[discount_rate_column].transform(_safe_zscore)

def build_feature_dataframe(
    df: pd.DataFrame,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Create the full order-level feature table for downstream modeling.

    Parameters
    ----------
    df:
        Raw DataFrame containing the raw supply-chain rows.
    rolling_window:
        Past-order window used for the rolling price average.
    """
    feature_frame = df.copy()

    feature_frame["price_deviation_from_supplier_category_avg"] = compute_price_deviation_by_supplier_category(
        feature_frame,
        rolling_window=rolling_window,
    )
    feature_frame["order_quantity_zscore_by_category"] = compute_order_quantity_zscore_by_category(
        feature_frame,
    )

    delivery_features = compute_delivery_delay_features(feature_frame)
    feature_frame["delivery_delay_days"] = delivery_features["delivery_delay_days"]
    feature_frame["delivery_delay_deviation"] = delivery_features["delivery_delay_deviation"]
    feature_frame["discount_rate_anomaly"] = compute_discount_rate_anomaly(feature_frame)

    return feature_frame

if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "Order Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Order Region": ["West", "West", "East", "East", "West"],
            "Category Name": ["Office", "Office", "Office", "Office", "Office"],
            "Order Item Product Price": [10, 12, 9, 11, 14],
            "Order Item Quantity": [5, 6, 7, 8, 9],
            "Days for shipping (real)": [4, 5, 6, 7, 8],
            "Days for shipment (scheduled)": [3, 3, 4, 4, 5],
            "Order Item Discount Rate": [0.05, 0.05, 0.10, 0.15, 0.20],
        }
    )

    feature_df = build_feature_dataframe(sample_df)
    print(feature_df[[
        "price_deviation_from_supplier_category_avg",
        "order_quantity_zscore_by_category",
        "delivery_delay_days",
        "delivery_delay_deviation",
        "discount_rate_anomaly",
    ]].head())
