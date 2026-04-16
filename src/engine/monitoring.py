"""Daily monitoring snapshot: actual vs backtest drift per model."""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

MONITORING_CSV = Path("data/trades/monitoring_daily.csv")

COLUMNS = [
    "date", "model",
    "n_trades_today", "bps_today",
    "cum_trades", "cum_bps", "cum_expected_bps", "cum_delta_pct", "cum_status",
    "window_7d_bps", "window_7d_expected_bps", "window_7d_delta_pct", "window_7d_status",
    "wallet_usd", "drawdown_pct",
]


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def snapshot(
    model: str,
    trades: Iterable[dict],
    daily_expected_bps: float,
    wallet_usd: float,
    drawdown_pct: float,
    session_start: datetime | None = None,
    days_elapsed: float | None = None,
    path: Path = MONITORING_CSV,
) -> None:
    """Upsert today's monitoring row for `model`. One row per (date, model).

    Pass either `session_start` (preferred) or explicit `days_elapsed` (for tests).
    """
    today = _today_utc()
    now = datetime.now(timezone.utc)
    cutoff_7d = now - timedelta(days=7)
    start_of_today = datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc)
    trades = list(trades)
    if days_elapsed is None:
        # Prefer earliest trade timestamp — survives bot restarts.
        earliest: datetime | None = None
        for t in trades:
            ts = _parse_ts(t.get("exit_time"))
            if ts is not None and (earliest is None or ts < earliest):
                earliest = ts
        if earliest is None and session_start is not None:
            if session_start.tzinfo is None:
                session_start = session_start.replace(tzinfo=timezone.utc)
            earliest = session_start
        if earliest is None:
            days_elapsed = 0.0
        else:
            days_elapsed = max((now - earliest).total_seconds() / 86400.0, 0.0)

    cum_trades = len(trades)
    cum_bps = sum((t.get("net_profit_bps") or 0.0) for t in trades)
    bps_today = 0.0
    n_today = 0
    bps_7d = 0.0
    for t in trades:
        ts = _parse_ts(t.get("exit_time"))
        if ts is None:
            continue
        bps = t.get("net_profit_bps") or 0.0
        if ts >= start_of_today:
            bps_today += bps
            n_today += 1
        if ts >= cutoff_7d:
            bps_7d += bps

    window_days = min(days_elapsed, 7.0)
    expected_7d = window_days * daily_expected_bps
    expected_cum = days_elapsed * daily_expected_bps

    def _delta_pct(actual: float, expected: float) -> float:
        if expected == 0 or days_elapsed < 1.0:
            return 0.0
        return round((actual - expected) / abs(expected) * 100.0, 1)

    def _status(pct: float, pending: bool) -> str:
        if pending:
            return "pending"
        p = abs(pct)
        if p <= 30:
            return "on_track"
        if p <= 60:
            return "drift"
        return "diverged"

    pending = days_elapsed < 1.0
    cum_delta = _delta_pct(cum_bps, expected_cum)
    win_delta = _delta_pct(bps_7d, expected_7d)

    row = {
        "date": today,
        "model": model,
        "n_trades_today": n_today,
        "bps_today": round(bps_today, 1),
        "cum_trades": cum_trades,
        "cum_bps": round(cum_bps, 1),
        "cum_expected_bps": round(expected_cum, 1),
        "cum_delta_pct": cum_delta,
        "cum_status": _status(cum_delta, pending),
        "window_7d_bps": round(bps_7d, 1),
        "window_7d_expected_bps": round(expected_7d, 1),
        "window_7d_delta_pct": win_delta,
        "window_7d_status": _status(win_delta, pending),
        "wallet_usd": round(wallet_usd, 2),
        "drawdown_pct": round(drawdown_pct, 4),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f)
                        if not (r.get("date") == today and r.get("model") == model)]
    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})
