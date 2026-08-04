from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/anomalies.db")

def init_db(db_path: str | Path = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            order_id TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            reason TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def insert_anomaly(
    timestamp: str,
    order_id: str | int,
    anomaly_score: float,
    reason: str,
    db_path: str | Path = DB_PATH,
) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO anomalies (timestamp, order_id, anomaly_score, reason)
        VALUES (?, ?, ?, ?)
        """,
        (timestamp, str(order_id), float(anomaly_score), reason),
    )
    conn.commit()
    conn.close()
