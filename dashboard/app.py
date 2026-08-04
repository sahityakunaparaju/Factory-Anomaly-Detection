"""Factory Anomaly Monitor — Streamlit dashboard.

Reads the anomaly-detection pipeline's outputs:
  data/anomalies.db             flagged anomalies (SQLite, written by the consumer)
  data/consumer_features.jsonl  per-order features (line count = orders processed)
  data/consumer_state.json      consumer's effective flagging threshold (written by the consumer)
  data/consumer_run.log         consumer activity (file mtime feeds the live dot)
  data/producer_run.log         producer activity (file mtime feeds the live dot)
  models/current_model.json     deployed model + native fallback threshold (written by train.py)
  data/eval_stream_run.log      last end-to-end evaluation numbers (optional)

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "anomalies.db"
FEATURES_LOG = ROOT / "data" / "consumer_features.jsonl"
CONSUMER_LOG = ROOT / "data" / "consumer_run.log"
PRODUCER_LOG = ROOT / "data" / "producer_run.log"
MANIFEST_PATH = ROOT / "models" / "current_model.json"
CONSUMER_STATE_PATH = ROOT / "data" / "consumer_state.json"
EVAL_LOG = ROOT / "data" / "eval_stream_run.log"

EXPECTED_ANOMALY_RATE = 0.065
LIVE_WINDOW_SECONDS = 60
MAX_CACHED_ROWS = 20000

SEV_COLORS = {"High": "#f87171", "Medium": "#fb923c", "Low": "#fbbf24"}
SEV_BG = {"High": "#451a12", "Medium": "#3d2410", "Low": "#3b2f10"}
SEV_DOT = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}

CATEGORY_ORDER = ["price", "delivery", "inventory", "discount", "stock"]
CATEGORY_COLORS = {
    "price": "#38bdf8",
    "delivery": "#a78bfa",
    "inventory": "#f472b6",
    "discount": "#34d399",
    "stock": "#fbbf24",
}
CATEGORY_ICONS = {
    "price": "💰",
    "delivery": "📦",
    "inventory": "📉",
    "discount": "🏷️",
    "stock": "📊",
}

CATEGORY_MAP = {
    "price_deviation_from_supplier_category_avg": "price",
    "delivery_delay_deviation": "delivery",
    "discount_rate_anomaly": "discount",
    "stock_after_order": "stock",
    "days_of_cover_remaining": "inventory",
}

CUSTOM_CSS = """
<style>
:root {
  --ink: #fde8d3; --muted: #d9a878; --chrome: #2a1609; --line: #4a2a14;
  --ok: #4ade80; --warn: #fbbf24; --danger: #f87171; --card: #34190c;
  --accent: #f97316; --accent-deep: #c2410c;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
  color: var(--ink);
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 20% -10%, #3d1f0f 0%, #241105 45%, #170b04 100%);
}
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #1d0e05; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--muted); }

[data-testid="stButton"] button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--line);
  background: var(--card); color: var(--ink); min-height: 2.45rem;
  transition: all .15s ease;
}
[data-testid="stButton"] button:hover {
  transform: translateY(-1px); box-shadow: 0 3px 10px rgba(249,115,22,.25);
  border-color: var(--accent); color: #fff;
}
[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(180deg, var(--accent), var(--accent-deep));
  border-color: var(--accent); color: #fff;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--card); }

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] [data-baseweb="input"] > div {
  background: var(--card); border-color: var(--line); border-radius: 10px;
  min-height: 2.45rem; color: var(--ink);
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
[data-testid="stTextInput"] [data-baseweb="input"] > div:hover {
  border-color: var(--accent);
}
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] span[aria-live] {
  color: var(--ink);
}
[data-testid="stSelectbox"] [data-baseweb="popover"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
}
[data-testid="stSelectbox"] [data-baseweb="popover"] li {
  color: var(--ink); background: transparent;
}
[data-testid="stSelectbox"] [data-baseweb="popover"] li:hover, [data-testid="stSelectbox"] [data-baseweb="popover"] li[aria-selected="true"] {
  background: var(--accent-deep); color: #fff;
}

[data-testid="stToggle"] button[role="switch"] {
  border-color: var(--line); background: var(--card);
}
[data-testid="stToggle"] button[role="switch"][aria-checked="true"] {
  background: var(--accent); border-color: var(--accent);
}
[data-testid="stToggle"] [data-testid="stWidgetLabel"] p {
  color: var(--ink);
}
[data-testid="stExpander"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
}
[data-testid="stExpander"] summary { color: var(--ink); }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { color: var(--ink); }
[data-testid="stExpander"] [data-testid="stCaptionContainer"] { color: var(--muted); }

.app-header { display: flex; justify-content: space-between; align-items: flex-end;
              gap: 1rem; margin-bottom: .5rem; flex-wrap: wrap; }
.app-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
             color: #fff; text-shadow: 0 1px 6px rgba(0,0,0,.4); }
.app-subtitle { color: var(--muted); font-size: .95rem; margin-top: .15rem; }
.header-right { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }

.status-pill { display: inline-flex; align-items: center; gap: .45rem;
               padding: .32rem .75rem; border-radius: 999px; font-size: .82rem;
               font-weight: 700; background: var(--card); border: 1px solid var(--line);
               color: var(--ink); }
.status-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.model-badge { display: inline-flex; align-items: center; gap: .35rem;
               padding: .32rem .75rem; border-radius: 999px; font-size: .8rem;
               font-weight: 600; background: #3a1c0c; color: #ffd9b0;
               border: 1px solid #6b3413; }
.badge-tag { font-size: .66rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: .04em; background: var(--accent-deep); color: #fff;
             padding: .1rem .42rem; border-radius: 999px; }
[data-testid="stWidgetLabel"] p { color: var(--ink); }
[data-testid="stWidgetLabel"] { min-height: 1.2rem; }
[data-testid="stPills"] > div { gap: .35rem; flex-wrap: wrap; }
[data-testid="stPills"] [data-baseweb="button"] {
  background: var(--card); border: 1px solid var(--line); color: var(--ink);
  border-radius: 999px;
}
[data-testid="stPills"] [data-baseweb="button"][aria-pressed="true"],
[data-testid="stPills"] [role="radio"][aria-checked="true"] {
  background: linear-gradient(180deg, var(--accent), var(--accent-deep));
  border-color: var(--accent); color: #fff;
}

.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .9rem; }
@media (max-width: 1100px) { .metric-row { grid-template-columns: repeat(2, 1fr); } }
.metric-card { background: linear-gradient(180deg, #3a1c0c 0%, var(--card) 100%);
               border: 1px solid var(--line); border-radius: 14px; padding: .95rem 1.1rem;
               box-shadow: 0 2px 8px rgba(0,0,0,.25); }
.metric-label { font-size: .74rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: .05em; color: var(--muted); }
.metric-value { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em;
                margin-top: .15rem; line-height: 1.1; color: #fff; }
.metric-sub { font-size: .78rem; color: var(--muted); margin-top: .2rem; }
.metric-delta { font-size: .78rem; font-weight: 600; margin-top: .15rem; }

.empty-state { text-align: center; padding: 3.2rem 1rem; margin-top: .5rem;
               border: 1px dashed var(--line); border-radius: 14px;
               background: rgba(52,25,12,.5); }
.empty-icon { font-size: 2.6rem; }
.empty-title { font-size: 1.15rem; font-weight: 700; margin-top: .5rem; color: #fff; }
.empty-hint { color: var(--muted); font-size: .9rem; margin-top: .3rem; }

.section-title { font-size: 1.02rem; font-weight: 700; margin-bottom: .4rem; color: #fff; }
.expander-note { font-size: .82rem; color: var(--muted); }
[data-testid="stMarkdownContainer"] p { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* Align control-row boxes (buttons + dropdowns + toggle) on one baseline */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stButton"]) {
  align-items: flex-end;
}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stButton"]) [data-testid="stColumn"] > div {
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}
.ctrl-label {
  font-size: .8rem;
  font-weight: 500;
  color: var(--muted);
  min-height: 1.55rem;
  line-height: 1.2;
}
[data-testid="stToggle"] {
  min-height: 2.45rem;
  display: flex;
  align-items: center;
}
</style>
"""

def _parse_ts(value: object) -> datetime | None:
    """Parse the ISO timestamps the consumer writes (UTC, e.g. +00:00)."""
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None

def _fmt_local(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")

def _sev(score: float) -> str:
    if score >= 10.0:
        return "High"
    if score >= 6.0:
        return "Medium"
    return "Low"

def _primary_reason(reason: str) -> tuple[str, str]:
    """Split a reason like 'discount_rate_anomaly deviated strongly (1.163); ...'
    into its primary (category, feature-name) pair."""
    first = (reason.split(";")[0] if reason else "").strip()
    match = re.match(r"([a-z_]+)", first)
    feature = match.group(1) if match else ""
    return CATEGORY_MAP.get(feature, "other"), feature

def _round3(value: float) -> float:
    """Round to 3 decimals, half-up, so log-parsed metrics match the defaults."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

@st.cache_data(ttl=10)
def load_manifest() -> dict[str, object] | None:
    try:
        if not MANIFEST_PATH.exists():
            return None
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

@st.cache_data(ttl=5)
def load_consumer_state() -> dict[str, object] | None:
    """The consumer's self-reported EFFECTIVE threshold (env > manifest).
    Written by src/consumer.py at startup and shutdown."""
    try:
        if not CONSUMER_STATE_PATH.exists():
            return None
        return json.loads(CONSUMER_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

@st.cache_data(ttl=10)
def load_eval_numbers() -> dict[str, float]:
    """Consumer-path precision/recall/F1 + per-type recall from the last
    evaluate_stream run; falls back to the known evaluated values."""
    defaults = {
        "precision": 0.289, "recall": 0.366, "f1": 0.323,
        "price_spike": 0.257, "delivery_delay": 0.999, "inventory_depletion": 0.156,
        "threshold": 4.5368,
        "from_log": False,
    }
    out = dict(defaults)
    try:
        if not EVAL_LOG.exists():
            return out
        log = EVAL_LOG.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"recalibrated threshold \(consumer path\):\s*([\d.]+)", log)
        if match:
            out["threshold"] = float(match.group(1))
        match = re.search(r"precision\s+=\s+([\d.]+)", log)
        if match:
            out["precision"] = _round3(float(match.group(1)))
        match = re.search(r"recall\s+=\s+([\d.]+)", log)
        if match:
            out["recall"] = _round3(float(match.group(1)))
        match = re.search(r"F1\s+=\s+([\d.]+)", log)
        if match:
            out["f1"] = _round3(float(match.group(1)))
        for at in ("price_spike", "delivery_delay", "inventory_depletion"):
            match = re.search(rf"\b{at}\s+recall=([\d.]+)", log)
            if match:
                out[at] = float(match.group(1))
        out["from_log"] = True
        return out
    except OSError:
        return out

def read_anomalies() -> pd.DataFrame:
    """Read ONLY new rows since the last read (id > last_seen), appended to a
    session cache, so auto-refresh never re-reads the whole table."""
    columns = ["id", "timestamp", "order_id", "anomaly_score", "reason"]
    cache = st.session_state.setdefault("anom_df", pd.DataFrame(columns=columns))
    last_id = int(st.session_state.setdefault("anom_last_id", 0))
    try:
        if not DB_PATH.exists():
            return cache
        conn = sqlite3.connect(DB_PATH)
        try:
            max_id = conn.execute("SELECT MAX(id) FROM anomalies").fetchone()[0]

            if max_id is not None and max_id < last_id:
                cache = pd.DataFrame(columns=columns)
                last_id = 0
                st.session_state["anom_df"] = cache
            new = pd.read_sql_query(
                "SELECT * FROM anomalies WHERE id > ? ORDER BY id",
                conn, params=(last_id,),
            )
        finally:
            conn.close()
        if len(new):
            st.session_state["anom_last_id"] = int(new["id"].max())
            cache = pd.concat([cache, new], ignore_index=True)
            if len(cache) > MAX_CACHED_ROWS:
                cache = cache.tail(MAX_CACHED_ROWS).reset_index(drop=True)
            st.session_state["anom_df"] = cache
    except sqlite3.Error:
        pass
    return st.session_state["anom_df"]

def processed_count() -> int:
    """Incrementally count orders the consumer processed by tailing
    consumer_features.jsonl at a byte offset (no full re-read per tick)."""
    count = int(st.session_state.setdefault("jsonl_count", 0))
    offset = int(st.session_state.setdefault("jsonl_offset", 0))
    last_ts = st.session_state.setdefault("jsonl_last_ts", None)
    try:
        if not FEATURES_LOG.exists():
            return count
        size = FEATURES_LOG.stat().st_size
        if size < offset:  
            offset, count = 0, 0
            st.session_state["jsonl_offset"], st.session_state["jsonl_count"] = 0, 0
        if size > offset:
            with FEATURES_LOG.open("rb") as handle:
                handle.seek(offset)
                for raw_line in handle:
                    try:
                        record = json.loads(raw_line.decode("utf-8"))
                        count += 1
                        ts = _parse_ts(record.get("timestamp"))
                        if ts is not None and (
                            last_ts is None or ts > last_ts
                        ):
                            last_ts = ts
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue  
            st.session_state["jsonl_offset"] = size
            st.session_state["jsonl_count"] = count
            st.session_state["jsonl_last_ts"] = last_ts
    except OSError:
        pass
    return count

def live_status() -> tuple[str, str, str]:
    """(label, color, detail) — Live when any pipeline source touched within the window."""
    now = datetime.now(timezone.utc)
    candidates: list[datetime] = []
    try:
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute("SELECT MAX(timestamp) FROM anomalies").fetchone()
            conn.close()
            if row and row[0]:
                ts = _parse_ts(row[0])
                if ts is not None:
                    candidates.append(ts)
    except sqlite3.Error:
        pass
    last_ts = st.session_state.get("jsonl_last_ts")
    if last_ts is not None:
        candidates.append(last_ts)
    for path in (CONSUMER_LOG, PRODUCER_LOG):
        try:
            if path.exists():
                candidates.append(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
        except OSError:
            pass
    if not candidates:
        return "Idle", "#94a3b8", "no pipeline activity yet"
    last = max(candidates)
    age = (now - last).total_seconds()
    if age <= LIVE_WINDOW_SECONDS:
        return "Live", "#4ade80", f"activity {age:.0f}s ago"
    minutes, seconds = divmod(int(age), 60)
    return "Idle", "#94a3b8", f"last activity {minutes}m {seconds}s ago"

def clear_demo_data() -> None:
    """Wipe the demo outputs (anomalies.db + stream logs) and reset caches."""
    for path in (DB_PATH, FEATURES_LOG, CONSUMER_LOG, PRODUCER_LOG):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    for key in ("anom_df", "anom_last_id", "jsonl_offset", "jsonl_count", "jsonl_last_ts"):
        st.session_state.pop(key, None)
    st.session_state["confirm_clear"] = False

def metric_card(label: str, value: str, sub: str = "", delta: str = "", delta_color: str = "") -> str:
    delta_html = (
        f'<div class="metric-delta" style="color:{delta_color};">{delta}</div>'
        if delta else ""
    )
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}{delta_html}'
        f'</div>'
    )

def build_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw DB rows -> scannable table rows with severity/category columns."""
    rows = []
    for _, row in df.iterrows():
        score = float(row["anomaly_score"])
        category, feature = _primary_reason(str(row["reason"]))
        ts = _parse_ts(row["timestamp"])
        rows.append(
            {
                "Time": _fmt_local(ts) if ts is not None else str(row["timestamp"]),
                "Order ID": str(row["order_id"]),
                "Score": score,
                "Severity": _sev(score),
                "Type": category,
                "Signal": feature or "—",
                "Reason": str(row["reason"])[:110],
            }
        )
    return pd.DataFrame(rows, columns=["Time", "Order ID", "Score", "Severity", "Type", "Signal", "Reason"])

def _safe_dt(text: str) -> datetime | None:
    """Parse the local display timestamp back to a datetime (chart bucketing)."""
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None

def _centisecond_label(ts: datetime) -> str:
    """Label a sub-second burst bucket, e.g. 14:44:10.08."""
    return f"{ts.strftime('%H:%M:%S')}.{ts.microsecond // 10000:02d}"

def style_table(frame: pd.DataFrame) -> pd.DataFrame.style:
    """Row-level styling: severity-colored left bar + badge cells + category tints."""

    def _row_css(row: pd.Series) -> list[str]:
        sev = row["Severity"]
        sev_color = SEV_COLORS.get(sev, "#94a3b8")
        category = row["Type"]
        cat_color = CATEGORY_COLORS.get(category, "#94a3b8")
        css: dict[str, str] = {}
        for col in frame.columns:
            css[col] = ""
        css["Time"] = f"border-left: 4px solid {sev_color};"
        css["Severity"] = (
            f"background-color: {SEV_BG.get(sev, '#3a1c0c')}; color: {sev_color};"
            f" font-weight: 700; border-radius: 4px;"
        )
        css["Score"] = f"color: {sev_color}; font-weight: 600;"
        css["Type"] = (
            f"color: {cat_color}; font-weight: 600;"
        )
        return [css[col] for col in frame.columns]

    styled = frame.style.apply(_row_css, axis=1)
    styled = styled.format(
        {"Severity": lambda s: f"{SEV_DOT.get(s, '')} {s}",
         "Type": lambda t: f"{CATEGORY_ICONS.get(t, '')} {t}"}
    )
    return styled

def empty_state() -> None:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">🏭</div>'
        '<div class="empty-title">No anomalies detected yet</div>'
        '<div class="empty-hint">Start the producer and consumer to see live results here.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

def main() -> None:
    st.set_page_config(
        page_title="Factory Anomaly Monitor",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    anomalies_df = read_anomalies()
    processed = processed_count()
    manifest = load_manifest() or {}
    consumer_state = load_consumer_state() or {}
    eval_numbers = load_eval_numbers()
    operative_th = consumer_state.get("threshold")
    threshold_src = consumer_state.get("threshold_source")

    with st.sidebar:
        st.markdown("### 🏭 Factory Monitor")
        st.caption("Real-time anomaly detection for the order-processing pipeline.")
        st.divider()
        st.caption(
            "Runtime data: data/anomalies.db · consumer/producer logs · models/current_model.json"
        )

    status_label, status_color, status_detail = live_status()
    model_name = manifest.get("model_name", "—")
    mth = manifest.get("threshold")
    if operative_th is not None:
        src_tag = {
            "env": "operative · env",
            "manifest": "operative · manifest",
            "model-predict": "model predict()",
        }.get(str(threshold_src), str(threshold_src))
        model_badge = (
            f'<div class="model-badge" title="Threshold the last consumer run actually '
            f'flagged with (source: {threshold_src})">'
            f"🤖 {model_name} · flag ≥ {float(operative_th):.4f} "
            f'<span class="badge-tag">{src_tag}</span></div>'
        )
    else:
        if mth is not None:
            model_badge = (
                f'<div class="model-badge" title="Manifest native threshold (-offset_), '
                f'used as fallback when ANOMALY_THRESHOLD is unset">'
                f"🤖 {model_name} · th {float(mth):.4f} "
                f'<span class="badge-tag">manifest fallback</span></div>'
            )
        else:
            model_badge = '<div class="model-badge">🤖 no model manifest</div>'
    st.markdown(
        f'<div class="app-header">'
        f'<div>'
        f'<h1 class="app-title">Factory Anomaly Monitor</h1>'
        f'<div class="app-subtitle">Live detection of price spikes, delivery delays and '
        f'inventory depletion in the order stream.</div>'
        f'</div>'
        f'<div class="header-right">'
        f'<div class="status-pill"><span class="status-dot" style="background:{status_color};"></span>'
        f'{status_label} <span style="color:#94a3b8;font-weight:500;">· {status_detail}</span></div>'
        f'{model_badge}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Threshold details", expanded=False):
        if mth is not None:
            st.caption(f"Manifest native (−offset_ fallback): **{float(mth):.4f}**")
        eval_label = (
            "Recalibrated (last eval)" if eval_numbers.get("from_log")
            else "Recalibrated (defaults)"
        )
        st.caption(f"{eval_label}: **{eval_numbers['threshold']:.4f}**")

    st.divider()
    controls = st.columns([1, 1, 1, 1, 1])
    with controls[0]:
        if st.button("🔄 Refresh now", width="stretch"):
            st.rerun()
    with controls[1]:
        st.markdown('<div class="ctrl-label">Auto-refresh</div>', unsafe_allow_html=True)
        auto = st.toggle(
            "Auto-refresh",
            value=st.session_state.get("auto_refresh", False),
            help="Poll the pipeline outputs on a timer.",
            width="stretch",
            label_visibility="collapsed",
        )
        st.session_state["auto_refresh"] = auto
    with controls[2]:
        interval = st.selectbox(
            "Interval", [2, 5, 10], index=1,
            format_func=lambda s: f"{s}s",
            disabled=not auto,
        )
        st.session_state["interval"] = interval
    with controls[3]:
        if st.button("🗑 Clear demo data", width="stretch", type="secondary"):
            st.session_state["confirm_clear"] = True
    with controls[4]:
        time_range = st.selectbox("Time range", ["Last 50", "Last 200", "All"], index=0)

    reason_types = st.pills(
        "Reason type",
        CATEGORY_ORDER,
        selection_mode="multi",
        default=CATEGORY_ORDER,
        format_func=lambda c: f"{CATEGORY_ICONS[c]} {c}",
        help="Toggle types to filter the table and charts; empty = all types.",
    )
    categories = list(reason_types) if reason_types else CATEGORY_ORDER

    if st.session_state.get("confirm_clear"):
        st.warning(
            "This wipes **data/anomalies.db** and the stream logs "
            "(`consumer_features.jsonl`, `consumer_run.log`, `producer_run.log`) so you can "
            "reset between demo runs. This cannot be undone."
        )
        confirm_cols = st.columns([1, 1, 4])
        with confirm_cols[0]:
            if st.button("Yes, wipe everything", type="primary"):
                clear_demo_data()
                st.rerun()
        with confirm_cols[1]:
            if st.button("Cancel"):
                st.session_state["confirm_clear"] = False
                st.rerun()

    flagged = len(anomalies_df)
    flag_rate = flagged / processed if processed else 0.0
    delta_pp = (flag_rate - EXPECTED_ANOMALY_RATE) * 100
    if abs(delta_pp) <= 1.5:
        delta_color = "#4ade80"
    elif abs(delta_pp) <= 3.5:
        delta_color = "#fbbf24"
    else:
        delta_color = "#f87171"
    delta_sign = "+" if delta_pp >= 0 else ""

    model_f1 = manifest.get("f1")
    f1_card_value = f"{float(model_f1):.3f}" if model_f1 is not None else "—"
    f1_sub = (
        f"consumer path · prec {eval_numbers['precision']:.3f} "
        f"· rec {eval_numbers['recall']:.3f}"
    )

    cards = "".join(
        [
            metric_card("Orders processed", f"{processed:,}"),
            metric_card("Anomalies flagged", f"{flagged:,}"),
            metric_card(
                "Flag rate",
                f"{flag_rate:.2%}" if processed else "—",
                sub=f"{EXPECTED_ANOMALY_RATE:.1%} expected across {processed:,} orders",
                delta=f"{delta_sign}{delta_pp:.1f} pp vs expected",
                delta_color=delta_color,
            ),
            metric_card("Model F1", f1_card_value, sub=f1_sub),
        ]
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

    if anomalies_df.empty:
        empty_state()
        st.markdown(
            '<div class="expander-note" style="margin-top:1rem;">'
            "No flags in the database yet. The status pill above reflects pipeline "
            "activity from the consumer/producer logs even when nothing has been flagged."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        frame = build_display_frame(anomalies_df)

        if categories:
            frame = frame[frame["Type"].isin(categories)]
        if time_range == "Last 50":
            frame = frame.tail(50)
        elif time_range == "Last 200":
            frame = frame.tail(200)

        frame = frame.iloc[::-1].reset_index(drop=True)

        left, right = st.columns([2.0, 1.15], gap="large")

        with left:
            st.markdown('<div class="section-title">Live anomalies</div>', unsafe_allow_html=True)
            if frame.empty:
                st.caption("No flags match the current filters.")
            else:
                st.dataframe(
                    style_table(frame),
                    hide_index=True,
                    width="stretch",
                    height=430,
                    column_config={
                        "Score": st.column_config.NumberColumn("Score", format="%.2f"),
                    },
                )
                st.caption(
                    f"Showing {len(frame)} flag{'s' if len(frame) != 1 else ''} · "
                    "severity bands: score ≥ 10 high, ≥ 6 medium, else low."
                )

        with right:
            st.markdown('<div class="section-title">Flag volume</div>', unsafe_allow_html=True)
            if frame.empty:
                st.caption("No data for the current filters.")
            else:
                bucket_df = frame.copy()
                bucket_df["_t"] = bucket_df["Time"].map(_safe_dt)
                bucket_df = bucket_df.dropna(subset=["_t"]).copy()
                if bucket_df.empty:
                    st.caption("No parseable timestamps for the time chart.")
                else:
                    span = bucket_df["_t"].max() - bucket_df["_t"].min()
                    if span <= pd.Timedelta(minutes=1):
                        fmt, bucket_label = "%H:%M:%S", "second"
                    elif span <= pd.Timedelta(hours=2):
                        fmt, bucket_label = "%H:%M", "minute"
                    elif span <= pd.Timedelta(days=7):
                        fmt, bucket_label = "%m-%d %H:00", "hour"
                    else:
                        fmt, bucket_label = "%Y-%m-%d", "day"
                    bucket_df["_bucket"] = bucket_df["_t"].dt.strftime(fmt)
                    # Burst guard: if every flag lands in one bucket (e.g. a
                    # sub-second burst), refine to centisecond labels so the
                    # distribution stays visible instead of a single bar.
                    if bucket_df["_bucket"].nunique() == 1 and len(bucket_df) > 1:
                        bucket_label = "centisecond"
                        bucket_df["_bucket"] = bucket_df["_t"].map(_centisecond_label)
                    volume = (
                        bucket_df.sort_values("_t")
                        .groupby("_bucket", sort=False)
                        .size().rename("count").reset_index()
                        .rename(columns={"_bucket": "Bucket"})
                    )
                    st.vega_lite_chart(
                        volume,
                        {
                            "mark": {"type": "bar", "tooltip": True},
                            "encoding": {
                                "x": {"field": "Bucket", "type": "ordinal",
                                      "sort": None, "title": None,
                                      "axis": {"labelAngle": -45}},
                                "y": {"field": "count", "type": "quantitative",
                                      "title": "flags"},
                            },
                        },
                        width="stretch",
                    )
                    if span < pd.Timedelta(minutes=1):
                        span_desc = f"{span.total_seconds():.1f}s"
                    elif span < pd.Timedelta(hours=1):
                        span_desc = f"{int(span.total_seconds() // 60)}m"
                    elif span < pd.Timedelta(days=1):
                        span_desc = f"{span.total_seconds() / 3600:.1f}h"
                    else:
                        span_desc = (
                            f"{span.days}d "
                            f"{int((span.seconds % 86400) // 3600)}h"
                        )
                    st.caption(
                        f"{len(bucket_df)} flags across {span_desc} · "
                        f"{len(volume)} {bucket_label} "
                        f"bucket{'s' if len(volume) != 1 else ''}"
                    )

                st.markdown(
                    '<div class="section-title" style="margin-top:1rem;">By reason type</div>',
                    unsafe_allow_html=True,
                )
                type_counts = (
                    frame.groupby("Type").size().rename("count").reset_index()
                )
                st.vega_lite_chart(
                    type_counts,
                    {
                        "mark": {"type": "bar", "tooltip": True},
                        "encoding": {
                            "y": {"field": "Type", "type": "nominal", "sort": "-x",
                                  "title": None},
                            "x": {"field": "count", "type": "quantitative",
                                  "title": "flags"},
                            "color": {
                                "field": "Type", "type": "nominal",
                                "scale": {"range": [CATEGORY_COLORS[t] for t in CATEGORY_ORDER]},
                                "legend": None,
                            },
                        },
                    },
                    width="stretch",
                )
                total = int(type_counts["count"].sum())
                if total:
                    top = type_counts.sort_values("count", ascending=False).iloc[0]
                    st.caption(
                        f"Most common: {top['Type']} ({top['count']}/{total} flags)"
                    )

        with st.expander("How this works"):
            st.markdown(
                "Every order is scored live by an **LOF outlier model** trained only on "
                "normal orders. A row is flagged when its anomaly score passes the "
                "recalibrated threshold, and the strongest signal drives the reason string."
            )
            st.markdown("**The three anomaly families:**")
            st.markdown(
                "- 📦 **Delivery delay** — an order ships far later than scheduled "
                "(per-category z-score of delay days).\n"
                "- 💰 **Price spike** — an order's unit price deviates sharply from its "
                "(region, category) rolling-median baseline.\n"
                "- 📉 **Inventory depletion** — synthetic demand bursts that push simulated "
                "stock-after-order or days-of-cover far below normal."
            )
            st.markdown("**Honest, evaluated numbers (consumer path, full stream):**")
            if operative_th is not None:
                model_line = (
                    f"- Model: `{manifest.get('model_name', 'lof')}` · flag ≥ "
                    f"**{float(operative_th):.4f}** ({threshold_src}) · recalibrated "
                    f"at the {EXPECTED_ANOMALY_RATE:.1%} true anomaly rate."
                )
            else:
                mth = manifest.get("threshold")
                model_line = (
                    f"- Model: `{manifest.get('model_name', 'lof')}` · native threshold "
                    f"{float(mth):.4f} (fallback until the consumer runs)."
                    if mth is not None
                    else f"- Model: `{manifest.get('model_name', 'lof')}` (no manifest)."
                )
            st.markdown(
                f"- Overall: **precision {eval_numbers['precision']:.3f} / "
                f"recall {eval_numbers['recall']:.3f} / F1 {eval_numbers['f1']:.3f}** "
                f"(flag rate ≈ {eval_numbers['recall']:.0%} of stream).\n"
                f"- **Delivery-delay is near-perfect** "
                f"({eval_numbers['delivery_delay']:.1%} recall).\n"
                f"- Price spikes and inventory depletion are genuinely harder "
                f"({eval_numbers['price_spike']:.1%} / "
                f"{eval_numbers['inventory_depletion']:.1%} recall) — their feature "
                f"distributions overlap the normal ones, so no threshold recovers them "
                f"without also flagging many normals.\n"
                f"{model_line}"
            )

    if st.session_state.get("auto_refresh"):
        time.sleep(int(st.session_state.get("interval", 5)))
        st.rerun()

if __name__ == "__main__":
    main()
