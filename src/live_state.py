"""Genuine live feature state for the streaming anomaly-detection consumer.

The consumer must compute model features from raw order fields alone. Every
feature is derived from a running state that is:

1. seeded once at startup from the historical (pre-streaming) dataset - the
   same pre-injection baseline the model was trained on - and
2. updated by every incoming order, so the NEXT order for the same group sees
   the updated history.

No precomputed feature value is ever taken from the payload.

State held per group
--------------------
* ``(Order Region, Category Name) -> RollingWindowState``
    The last 30 order prices in chronological order, used to derive the frozen
    rolling-median baseline. Mirrors ``compute_price_baseline_reference()`` in
    src/features.py: the baseline for an order is the median of the *previous*
    up-to-30 prices in the group (shift-1 semantics), so
    ``price_deviation_from_supplier_category_avg`` matches training exactly.

* ``Category Name -> FrozenZScoreState``
    Mean/std (ddof=0) of delivery delays and discount rates over the
    pre-injection baseline, mirroring the full-sample per-category z-scores the
    training pipeline computes. z = (value - mean) / std, or 0 when std is 0.

* ``(Product Card Id, Order Region) -> InventoryState``
    Frozen base stock, 14-day restock windows, quantity consumed in the current
    window and the last 7 demand observations. Mirrors
    ``simulate_inventory()`` in src/inject_anomalies.py and is seeded from the
    same ``inventory_pair_constants()`` so values match training exactly.

Caveats
-------
* Cold start: an order for a (region, category) or (product, region) group
  with no seeded history falls back to a zero baseline / zero base stock
  (deviation 0, negative stock), mirroring the previous consumer behaviour.
* Replay order: the live features match the offline pipeline exactly when
  orders arrive in chronological order (verified by
  src/verify_live_features.py) and for the FIRST order of each group in any
  order. With a shuffled replay, later orders of a group see state updated by
  earlier (out-of-order) arrivals, so their values approximate the offline
  values rather than matching them exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
import pandas as pd

from src.features import _resolve_supplier_column
from src.inject_anomalies import inventory_pair_constants

PRICE_COLUMN = "Order Item Product Price"
CATEGORY_COLUMN = "Category Name"
QUANTITY_COLUMN = "Order Item Quantity"
REGION_COLUMN = "Order Region"
PRODUCT_COLUMN = "Product Card Id"
SHIPPING_REAL_COLUMN = "Days for shipping (real)"
SHIPPING_SCHEDULED_COLUMN = "Days for shipment (scheduled)"
DISCOUNT_RATE_COLUMN = "Order Item Discount Rate"
ORDER_DATE_COLUMN = "order date (DateOrders)"
ORDER_ITEM_ID_COLUMN = "Order Item Id"

PRICE_WINDOW = 30  
DEMAND_WINDOW = 7  
RESTOCK_INTERVAL_DAYS = 14  
BASE_STOCK_MULTIPLE = 2  

FEATURE_COLUMNS = [
    "price_deviation_from_supplier_category_avg",
    "delivery_delay_deviation",
    "discount_rate_anomaly",
    "stock_after_order",
    "days_of_cover_remaining",
]

@dataclass
class RollingWindowState:
    """Last ``PRICE_WINDOW`` values of a group, in arrival (chronological) order."""

    history: Deque[float] = field(default_factory=lambda: deque(maxlen=PRICE_WINDOW))

    def baseline(self) -> float:
        """Median of the window; 0.0 when empty (cold start handled by caller)."""
        if not self.history:
            return 0.0
        window = list(self.history)
        return float(np.median(window[-min(PRICE_WINDOW, len(window)):]))

    def append(self, value: float) -> None:
        if np.isfinite(value):
            self.history.append(float(value))

@dataclass
class FrozenZScoreState:
    """Frozen full-sample statistics of a category over the pre-injection baseline."""

    mean: float = 0.0
    std: float = 0.0

    def z(self, value: float) -> float:
        if not np.isfinite(value) or self.std <= 0 or not np.isfinite(self.std):
            return 0.0
        return (value - self.mean) / self.std

@dataclass
class InventoryState:
    """Simulated stock for one (product, region) pair, updated per order.

    Mirrors ``simulate_inventory()``: stock starts at the frozen base stock and
    is reduced by each order's quantity, with the counter reset whenever the
    order date crosses into a new 14-day restock window. ``days_of_cover``
    divides the remaining stock by the rolling mean of the last 7 quantities
    (falling back to the current quantity when history is empty).
    """

    base_stock: float = 0.0
    first_order_date: pd.Timestamp | None = None
    restock_interval_days: int = RESTOCK_INTERVAL_DAYS
    window_id: int = 0
    consumed_in_window: float = 0.0
    demand_history: Deque[float] = field(default_factory=lambda: deque(maxlen=DEMAND_WINDOW))

    def update(self, quantity: float, order_date: object) -> tuple[float, float]:
        """Apply one order; return (stock_after_order, days_of_cover_remaining)."""
        if self.first_order_date is not None and pd.notna(order_date):
            new_window = int(
                (pd.Timestamp(order_date) - self.first_order_date).days // self.restock_interval_days
            )
            if new_window > self.window_id:
                self.window_id = new_window
                self.consumed_in_window = 0.0

        quantity = 0.0 if pd.isna(quantity) else float(quantity)
        self.consumed_in_window += quantity
        stock_after = self.base_stock - self.consumed_in_window

        mean_demand = float(np.mean(list(self.demand_history))) if self.demand_history else float(quantity)
        if not np.isfinite(mean_demand) or mean_demand <= 0:
            mean_demand = max(float(quantity), 1.0)
        days_of_cover = stock_after / mean_demand if mean_demand > 0 else np.nan

        self.demand_history.append(float(quantity))
        if np.isfinite(days_of_cover):
            return stock_after, float(days_of_cover)
        return stock_after, np.nan

@dataclass
class LiveFeatureState:
    """All per-group state used to compute features live from raw fields."""

    price_state: dict[tuple[str, str], RollingWindowState] = field(default_factory=dict)
    delivery_stats: dict[str, FrozenZScoreState] = field(default_factory=dict)
    discount_stats: dict[str, FrozenZScoreState] = field(default_factory=dict)
    inventory_state: dict[tuple[str, str], InventoryState] = field(default_factory=dict)

    def seed_from_historical(
        self,
        history_df: pd.DataFrame,
        exclude_order_item_ids: set[str] | None = None,
    ) -> None:
        """Seed the state from the historical (pre-injection) dataset.

        ``exclude_order_item_ids`` optionally excludes a set of order rows that
        will later be replayed as the "stream". Those rows are excluded from the
        *causal* seeds (price windows, window consumption, demand history) so
        that when they are processed live the state is exactly what the offline
        pipeline had before those rows. Frozen group constants and per-category
        statistics are always computed from the full dataset, matching the
        training baseline.
        """
        full = history_df.copy()
        full[ORDER_DATE_COLUMN] = pd.to_datetime(full[ORDER_DATE_COLUMN], errors="coerce")
        full = full.sort_values(ORDER_DATE_COLUMN, kind="mergesort").reset_index(drop=True)

        exclude: set[str] = set(exclude_order_item_ids or [])

        stream_mask = (
            full[ORDER_ITEM_ID_COLUMN].astype(str).isin(exclude) if exclude else None
        )
        seed_rows = full[~stream_mask] if stream_mask is not None else full
        seed_groups = (
            seed_rows.groupby([PRODUCT_COLUMN, REGION_COLUMN], dropna=False)
            if stream_mask is not None
            else None
        )
        seed_group_lookup = seed_groups.groups if seed_groups is not None else None

        supplier_column = _resolve_supplier_column(seed_rows)
        for (region, category), group in seed_rows.groupby(
            [supplier_column, CATEGORY_COLUMN], dropna=False
        ):
            window = RollingWindowState()
            for price in group[PRICE_COLUMN]:
                window.append(float(price))
            self.price_state[(str(region), str(category))] = window

        delivery_delay = pd.to_numeric(full[SHIPPING_REAL_COLUMN], errors="coerce") - pd.to_numeric(
            full[SHIPPING_SCHEDULED_COLUMN], errors="coerce"
        )
        discount_rate = pd.to_numeric(full[DISCOUNT_RATE_COLUMN], errors="coerce")
        for category, group in full.groupby(CATEGORY_COLUMN, dropna=False):
            delay = delivery_delay.loc[group.index].dropna().astype(float)
            self.delivery_stats[str(category)] = FrozenZScoreState(
                mean=float(delay.mean()) if len(delay) else 0.0,
                std=float(delay.std(ddof=0)) if len(delay) > 1 else 0.0,
            )
            discount = discount_rate.loc[group.index].dropna().astype(float)
            self.discount_stats[str(category)] = FrozenZScoreState(
                mean=float(discount.mean()) if len(discount) else 0.0,
                std=float(discount.std(ddof=0)) if len(discount) > 1 else 0.0,
            )

        constants = inventory_pair_constants(
            full,
            restock_interval_days=RESTOCK_INTERVAL_DAYS,
            base_stock_multiple=BASE_STOCK_MULTIPLE,
        )
        for (product, region), group in full.groupby([PRODUCT_COLUMN, REGION_COLUMN], dropna=False):
            pair_key = f"{str(product)} | {str(region)}"
            if pair_key not in constants.index:
                continue
            constant_row = constants.loc[pair_key]
            inventory = InventoryState(
                base_stock=float(constant_row["base_stock"]),
                first_order_date=pd.Timestamp(constant_row["first_order_date"]),
                restock_interval_days=RESTOCK_INTERVAL_DAYS,
            )

            pair_seed = group
            if seed_group_lookup is not None:
                if (product, region) in seed_group_lookup:
                    pair_seed = seed_groups.get_group((product, region))
                else:
                    pair_seed = seed_rows.iloc[0:0]
            if len(pair_seed):
                dates = pd.to_datetime(pair_seed[ORDER_DATE_COLUMN])
                windows = (
                    (dates - inventory.first_order_date).dt.days // inventory.restock_interval_days
                ).astype(int)
                last_window = int(windows.max())
                inventory.window_id = last_window
                inventory.consumed_in_window = float(
                    pair_seed.loc[windows == last_window, QUANTITY_COLUMN].astype(float).sum()
                )
                inventory.demand_history = deque(
                    pair_seed[QUANTITY_COLUMN].astype(float).tolist()[-DEMAND_WINDOW:],
                    maxlen=DEMAND_WINDOW,
                )

            self.inventory_state[(str(product), str(region))] = inventory

    def compute_features(self, row: dict[str, object]) -> dict[str, float]:
        """Compute the 5 model features for one raw order, updating the state.

        Order matters: every feature is computed from the state BEFORE this
        order is applied, then the state is advanced so the next order for the
        same group uses the updated history.
        """
        region = str(row.get(REGION_COLUMN, "unknown"))
        category = str(row.get(CATEGORY_COLUMN, "unknown"))
        product = str(row.get(PRODUCT_COLUMN, "unknown"))

        price_window = self.price_state.setdefault((region, category), RollingWindowState())
        raw_price = row.get(PRICE_COLUMN)
        current_price = (
            float(pd.to_numeric(raw_price, errors="coerce")) if pd.notna(raw_price) else float("nan")
        )
        baseline = price_window.baseline() if price_window.history else current_price
        price_deviation = current_price - baseline
        price_window.append(current_price)

        raw_real = row.get(SHIPPING_REAL_COLUMN)
        raw_scheduled = row.get(SHIPPING_SCHEDULED_COLUMN)
        delivery_delay = float(pd.to_numeric(raw_real, errors="coerce")) - float(
            pd.to_numeric(raw_scheduled, errors="coerce")
        )
        delivery_z = self.delivery_stats.setdefault(category, FrozenZScoreState()).z(delivery_delay)

        raw_discount = row.get(DISCOUNT_RATE_COLUMN)
        discount_rate = (
            float(pd.to_numeric(raw_discount, errors="coerce")) if pd.notna(raw_discount) else float("nan")
        )
        discount_z = self.discount_stats.setdefault(category, FrozenZScoreState()).z(discount_rate)

        inventory = self.inventory_state.setdefault(
            (product, region),
            InventoryState(base_stock=0.0),
        )
        raw_quantity = row.get(QUANTITY_COLUMN)
        quantity = (
            float(pd.to_numeric(raw_quantity, errors="coerce")) if pd.notna(raw_quantity) else 0.0
        )
        stock_after_order, days_of_cover = inventory.update(quantity, row.get(ORDER_DATE_COLUMN))

        return {
            "price_deviation_from_supplier_category_avg": float(price_deviation),
            "delivery_delay_deviation": float(delivery_z),
            "discount_rate_anomaly": float(discount_z),
            "stock_after_order": float(stock_after_order),
            "days_of_cover_remaining": 0.0 if not np.isfinite(days_of_cover) else float(days_of_cover),
        }
