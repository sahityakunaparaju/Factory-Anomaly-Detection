"""Offline reference feature computation shared by the verification tooling.

The streaming consumer computes features live from raw order fields; these
functions compute the same features in batch over the full pre-injection
baseline so live output can be compared against the offline pipeline without
depending on the consumer / Kafka stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    compute_delivery_delay_features,
    compute_discount_rate_anomaly,
    compute_price_deviation_by_supplier_category,
)
from src.inject_anomalies import simulate_inventory

ORDER_ITEM_ID = "Order Item Id"
ORDER_DATE_COLUMN = "order date (DateOrders)"

def build_offline_reference(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the offline reference for every row on the pre-injection baseline.

    The offline price-baseline function sorts the frame internally by order
    date and resets its index, so its output is only aligned with the input
    when the input is pre-sorted the same way. We therefore sort first and
    attach every reference value back to its row by Order Item Id.
    """

    sorted_raw = raw.copy()
    sorted_raw[ORDER_DATE_COLUMN] = pd.to_datetime(
        sorted_raw[ORDER_DATE_COLUMN], errors="coerce"
    )
    sorted_raw = sorted_raw.dropna(subset=[ORDER_DATE_COLUMN]).sort_values(
        ORDER_DATE_COLUMN, kind="mergesort"
    ).reset_index(drop=True)
    ref = sorted_raw.copy()

    price_dev = compute_price_deviation_by_supplier_category(sorted_raw)
    ref = ref.merge(
        pd.DataFrame(
            {
                ORDER_ITEM_ID: sorted_raw[ORDER_ITEM_ID].values,
                "ref_price_deviation": np.asarray(price_dev, dtype=float),
            }
        ),
        on=ORDER_ITEM_ID,
        how="left",
    )
    ref = ref.merge(
        simulate_inventory(sorted_raw)[
            [ORDER_ITEM_ID, "stock_after_order", "days_of_cover_remaining"]
        ],
        on=ORDER_ITEM_ID,
        how="left",
    )
    ref = ref.merge(
        compute_delivery_delay_features(sorted_raw)[[ORDER_ITEM_ID, "delivery_delay_deviation"]],
        on=ORDER_ITEM_ID,
        how="left",
    )
    ref = ref.merge(
        pd.DataFrame(
            {
                ORDER_ITEM_ID: sorted_raw[ORDER_ITEM_ID].values,
                "discount_rate_anomaly": np.asarray(
                    compute_discount_rate_anomaly(sorted_raw), dtype=float
                ),
            }
        ),
        on=ORDER_ITEM_ID,
        how="left",
    )
    return ref.set_index(ORDER_ITEM_ID)
