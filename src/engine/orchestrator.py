"""Orchestrator — owns the SHARED world and gives each track its turn.

The successor to the old monolithic bot.py Alpha class. Everything model-
specific lives in the tracks (track.py); everything here is shared:
  - Binance WebSocket connector + candle buffer
  - indicator computation (once per bar, consumed by all tracks)
  - global bar counter
  - dashboard bridge + chart history + proximity display
  - monitoring snapshots (expected-vs-actual daily rows)

build_tracks() is the ONE place a new model is added to the live system.

Run flow:  bot.py main() -> build_tracks() -> Orchestrator -> start()
"""

import asyncio
import logging
import signal
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from . import monitoring as _monitoring
from .binance_connector import BinanceConnector
from .config.constants import SYMBOL, TIMEFRAME
from .config.schema import AppConfig
from .dashboard_bridge import DashboardBridge
from .persistence import StateStore, TradeLog, paths_for
from .risk.decision_logger import DecisionLogger
from .signals.direction_attention import DirectionAttention
from .signals.ml_v1 import MLV1
from .signals.ml_v3 import MLV3
from .strategy import V12Strategy
from .track import (
    KIND_TRADE_CLOSED,
    MLSource,
    MLV3Source,
    StrategyTrack,
    V14Source,
)

logger = logging.getLogger(__name__)

# Project root: src/engine/orchestrator.py -> parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Need 200+ bars for SMA200 warm-up + margin
MIN_BUFFER_SIZE = 300
BUFFER_SIZE = 500


@dataclass(frozen=True)
class TrackMeta:
    """Display + monitoring metadata for one track (consumed by the bridge)."""
    title: str
    accent_color: str
    daily_expected_bps: float
    sort_order: int
    monitoring_name: Optional[str]  # key for monitoring.py rows (legacy names kept)


# Registration order = dashboard card order. Adding model #5 = one entry
# here + its params.yaml block + model files.
TRACK_METAS = {
    "V_RULE_BASED": TrackMeta("Rule Based (V1.4)", "#3b82f6", 0.0, 0, None),
    "ML_V1":        TrackMeta("ML V1",             "#8b5cf6", 7.38, 1, "ML_V1"),
    "ML_ATTN":      TrackMeta("ML V2 Attention",   "#f59e0b", 10.02, 2, "ML_V2_ATTENTION"),
    "ML_V3":        TrackMeta("ML V3",             "#10b981", 27.5, 3, "ML_V3"),
}


def _load_exit_versions() -> dict:
    """Per-model exit versions from params.yaml (single source of truth)."""
    with open(PROJECT_ROOT / "configs" / "params.yaml") as f:
        p = yaml.safe_load(f)
    return {
        "V_RULE_BASED": p.get("backtest", {}).get("exit_version", "v1"),
        "ML_V1": p.get("ml_v1", {}).get("inference", {}).get("exit_version", "v1"),
        "ML_ATTN": p.get("ml_v2_attention", {}).get("inference", {}).get("exit_version", "v1"),
        "ML_V3": p.get("ml_v3", {}).get("inference", {}).get("exit_version", "v1"),
    }


def build_tracks(config: AppConfig, trades_dir: Optional[Path] = None) -> list:
    """Construct the four strategy tracks. THE place to add model #5.

    Args:
        config: shared AppConfig.
        trades_dir: base dir for CSVs/state (default: data/trades).
                    Injectable so tests never touch real files.
    """
    trades_dir = Path(trades_dir) if trades_dir else PROJECT_ROOT / "data" / "trades"
    mode = config.execution.mode
    ev = _load_exit_versions()

    specs = [
        # (name, source, csv_tag, allow_same_bar_entry)
        ("V_RULE_BASED", V14Source(V12Strategy(config)), config.config_hash(), False),
        ("ML_V1", MLSource(MLV1(
            model_path=PROJECT_ROOT / "models" / "ML_V1" / "direction_model.onnx",
            scaler_path=PROJECT_ROOT / "models" / "ML_V1" / "scaler.npz",
        )), "ml_v1", True),
        ("ML_ATTN", MLSource(DirectionAttention(
            model_path=PROJECT_ROOT / "models" / "ML_V2_ATTENTION" / "attention_model.onnx",
            scaler_path=PROJECT_ROOT / "models" / "ML_V2_ATTENTION" / "scaler.npz",
        )), "ml_attn_v1", True),
        ("ML_V3", MLV3Source(MLV3(
            model_path=PROJECT_ROOT / "models" / "ML_V3" / "v3_model.onnx",
            scaler_path=PROJECT_ROOT / "models" / "ML_V3" / "scaler.npz",
        )), config.config_hash(), True),
    ]

    tracks = []
    for name, source, csv_tag, same_bar in specs:
        paths = paths_for(name, mode, trades_dir)
        tracks.append(StrategyTrack(
            name=name,
            source=source,
            config=config,
            exit_version=ev[name],
            trade_log=TradeLog(paths.trades_csv),
            state_store=StateStore(paths.state_json),
            decision_logger=DecisionLogger(log_dir=str(paths.decision_log_dir)),
            csv_tag=csv_tag,
            allow_same_bar_entry=same_bar,
        ))
    logger.info("Exit versions: %s", ev)
    return tracks


class Orchestrator:
    """Shared world + per-track turn-taking. See module docstring."""

    def __init__(self, config: AppConfig, tracks: list):
        self.cfg = config
        self.tracks = tracks
        self.strategy = V12Strategy(config)  # shared indicators + proximity thresholds

        self._bar_count = 0
        self._running = False
        self._connector = None
        self._initial_push_done = False
        self._chart_history_loaded = False

        # Dashboard (optional — attach_dashboard())
        self.state = None
        self.bridge: Optional[DashboardBridge] = None

    # ------------------------------------------------------------ dashboard

    def attach_dashboard(self, state) -> None:
        """Wire the dashboard: config, per-track cards, restart replay."""
        self.state = state
        self.bridge = DashboardBridge(state)

        state.set_config(
            pair=SYMBOL,
            timeframe=TIMEFRAME,
            mode=self.cfg.execution.mode,
            max_bars=self.cfg.exit.v1_max_bars,
            reentry_enabled=self.cfg.reentry.enabled,
            config_hash=self.cfg.config_hash(),
        )
        for track in self.tracks:
            meta = TRACK_METAS.get(track.name)
            if meta is None:
                meta = TrackMeta(track.name, "#8892a0", 0.0, 99, None)
            self.bridge.register(
                track,
                title=meta.title,
                accent_color=meta.accent_color,
                daily_expected_bps=meta.daily_expected_bps,
                sort_order=meta.sort_order,
            )
            # Replay history so the dashboard survives restarts
            try:
                self.bridge.handle_all(track, track.restore())
            except Exception as e:
                logger.error("[%s] restore replay failed: %s", track.name, e)

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        logger.info(
            "Starting ALPHA | mode=%s | hash=%s | %d tracks",
            self.cfg.execution.mode, self.cfg.config_hash(), len(self.tracks),
        )
        self._running = True
        self._connector = BinanceConnector(
            symbol=SYMBOL,
            interval=TIMEFRAME,
            buffer_size=BUFFER_SIZE,
            on_candle_close=self._on_candle_close,
            on_candle_update=self._on_candle_update,
        )

        if self.state:
            self.state.start()
        # Yield so the web server task can bind its port
        await asyncio.sleep(0.1)

        # Chart history load is blocking I/O — keep the event loop responsive
        if self.state:
            await asyncio.get_event_loop().run_in_executor(None, self._load_chart_history)

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass  # Windows

        try:
            await self._connector.start()
        except KeyboardInterrupt:
            await self.stop()

    async def stop(self) -> None:
        logger.info("Stopping ALPHA...")
        self._running = False
        if self._connector:
            await self._connector.stop()
        if self.state:
            self.state.stop()
        for track in self.tracks:
            s = track.stats()
            if s.total_trades:
                logger.info(
                    "[%s] session: %d trades, %.1f%% win, %+.0f bps",
                    track.name, s.total_trades, s.win_rate * 100, s.total_bps,
                )

    # ------------------------------------------------------------ ticks

    async def _on_candle_update(self, candle: dict) -> None:
        """Every WebSocket tick: live price + per-track price exits."""
        current_price = candle["close"]
        tick_time = candle.get("close_time") or candle.get("open_time") or datetime.now(timezone.utc)

        if self.state:
            if not self._initial_push_done:
                df = self._connector.get_candle_buffer()
                if len(df) < MIN_BUFFER_SIZE:
                    return
                if not self._chart_history_loaded:
                    # Chart fetch failed — set indicators from live buffer
                    df = self.strategy.compute_indicators(df)
                    latest = df.iloc[-1]
                    self.state.update_status(
                        price=current_price,
                        regime="BULL" if latest.get("bull_market", False) else "BEAR",
                        bar_count=self._bar_count,
                        rsi=round(float(latest.get("rsi", 0)), 1),
                        atr_percentile=round(float(latest.get("atr_percentile", 0)), 1),
                        ema_separation=round(float(latest.get("ema_separation", 0)), 4),
                    )
                else:
                    self.state.update_status(price=current_price, bar_count=self._bar_count)
                self._initial_push_done = True
                logger.info("Dashboard initialized from buffer (%d bars)", len(df))
                return

            self.state.update_status(price=current_price)
            self.state.update_current_candle({
                "time": int(candle["open_time"].timestamp()),
                "open": round(candle["open"], 2),
                "high": round(candle["high"], 2),
                "low": round(candle["low"], 2),
                "close": round(current_price, 2),
                "volume": round(candle.get("volume", 0), 2),
            })

        # Price-based exits for every track (dashboard optional)
        for track in self.tracks:
            events = track.on_tick(current_price, tick_time)
            if self.bridge:
                self.bridge.handle_all(track, events, price=current_price)
            self._snapshot_closed(track, events)

    # ------------------------------------------------------------ bar close

    async def _on_candle_close(self, candle: dict) -> None:
        """Every 15m close: shared indicators once, then each track's turn."""
        self._bar_count += 1

        df = self._connector.get_candle_buffer()
        if len(df) < MIN_BUFFER_SIZE:
            logger.debug("Warming up: %d/%d bars", len(df), MIN_BUFFER_SIZE)
            return

        df = self.strategy.compute_indicators(df)
        latest = df.iloc[-1]
        current_price = candle["close"]

        if self.state:
            self.state.update_status(
                price=current_price,
                regime="BULL" if latest.get("bull_market", False) else "BEAR",
                bar_count=self._bar_count,
                rsi=round(float(latest.get("rsi", 0)), 1),
                atr_percentile=round(float(latest.get("atr_percentile", 0)), 1),
                ema_separation=round(float(latest.get("ema_separation", 0)), 4),
            )
            self.state.close_candle({
                "time": int(df.index[-1].timestamp()),
                "open": round(float(latest["open"]), 2),
                "high": round(float(latest["high"]), 2),
                "low": round(float(latest["low"]), 2),
                "close": round(float(latest["close"]), 2),
                "volume": round(float(latest.get("volume", 0)), 2),
                "sma": round(float(latest["sma"]), 2) if pd.notna(latest.get("sma")) else None,
                "rsi": round(float(latest["rsi"]), 1) if pd.notna(latest.get("rsi")) else None,
            })
            self.state.update_proximity(self._compute_proximity(latest))

        for track in self.tracks:
            try:
                events = track.on_bar_close(candle, df, self._bar_count)
            except Exception as e:
                logger.error("[%s] on_bar_close failed: %s", track.name, e, exc_info=True)
                continue
            if self.bridge:
                self.bridge.handle_all(track, events, price=current_price)
            self._snapshot_closed(track, events)

    # ------------------------------------------------------------ monitoring

    def _snapshot_closed(self, track: StrategyTrack, events: list) -> None:
        """Write the daily expected-vs-actual row when a live trade closes."""
        meta = TRACK_METAS.get(track.name)
        if meta is None or meta.monitoring_name is None or self.state is None:
            return
        if not any(e.kind == KIND_TRADE_CLOSED and not e.restored for e in events):
            return
        try:
            _monitoring.snapshot(
                model=meta.monitoring_name,
                trades=[asdict(t) for t in track.pm.trades],
                daily_expected_bps=meta.daily_expected_bps,
                wallet_usd=track.wallet,
                drawdown_pct=track.health.get_drawdown_pct(track.wallet),
                session_start=self.state.get_session_start(),
            )
        except Exception as e:
            logger.warning("Monitoring snapshot failed for %s: %s", track.name, e)

    # ------------------------------------------------------------ chart history
    # (moved unchanged from old bot.py)

    def _load_chart_history(self) -> None:
        """Load 15m candle history (Binance REST, TimescaleDB fallback)."""
        if not self.state:
            return
        df = self._fetch_chart_from_binance()
        if df is None:
            df = self._fetch_chart_from_db()
        if df is None:
            return

        df = self.strategy.compute_indicators(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        cutoff = df.index.max() - pd.Timedelta(days=11)
        df = df[df.index >= cutoff]

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": int(idx.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 2),
                "sma": round(float(row["sma"]), 2) if pd.notna(row.get("sma")) else None,
                "rsi": round(float(row["rsi"]), 1) if pd.notna(row.get("rsi")) else None,
            })
        self.state.set_candle_history(candles)

        last_row = df.iloc[-1]
        self.state.update_status(
            price=round(float(last_row["close"]), 2),
            regime="BULL" if last_row.get("bull_market", False) else "BEAR",
            rsi=round(float(last_row["rsi"]), 1) if pd.notna(last_row.get("rsi")) else 0,
            atr_percentile=round(float(last_row["atr_percentile"]), 1) if pd.notna(last_row.get("atr_percentile")) else 0,
            ema_separation=round(float(last_row["ema_separation"]), 4) if pd.notna(last_row.get("ema_separation")) else 0,
        )
        self.state.update_proximity(self._compute_proximity(last_row))
        self._chart_history_loaded = True
        logger.info("Loaded %d 15m candles for chart display", len(candles))

    def _fetch_chart_from_binance(self) -> Optional[pd.DataFrame]:
        import requests as req
        logger.info("Loading chart history from Binance REST API...")
        try:
            resp = req.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": SYMBOL, "interval": TIMEFRAME, "limit": 1500},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            logger.warning("Binance API chart fetch failed: %s", e)
            return None
        if not raw:
            return None

        rows = [{
            "time": pd.Timestamp(k[0], unit="ms"),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
        } for k in raw]
        df = pd.DataFrame(rows).set_index("time")
        # Drop the currently forming (incomplete) candle
        if len(df) > 1:
            df = df.iloc[:-1]
        logger.info("Fetched %d native 15m candles from Binance", len(df))
        return df

    def _fetch_chart_from_db(self) -> Optional[pd.DataFrame]:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not set, skipping DB chart fallback")
            return None
        logger.info("Falling back to TimescaleDB for chart history...")
        try:
            engine = create_engine(db_url, connect_args={"connect_timeout": 10})
            query = text("""
                SELECT
                    time_bucket('15 minutes', time) AS time,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume
                FROM ohlcv_data
                WHERE pair = :pair
                  AND time >= NOW() - INTERVAL '14 days'
                GROUP BY 1
                ORDER BY 1
            """)
            df = pd.read_sql(query, engine, params={"pair": SYMBOL})
            engine.dispose()
        except Exception as e:
            logger.error("DB chart fetch failed: %s", e)
            return None
        if df.empty:
            return None
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.index = df.index.tz_localize(None)
        return df

    # ------------------------------------------------------------ proximity

    def _compute_proximity(self, latest: pd.Series) -> dict:
        """Signal proximity for the 4 rule-based signal types (display)."""
        c = self.cfg
        rsi = float(latest.get("rsi", 50))
        atr_pctl = float(latest.get("atr_percentile", 0))
        ema_sep = float(latest.get("ema_separation", 0))
        is_bull = bool(latest.get("bull_market", False))

        return {
            "V12_LONG": {
                "direction": "LONG",
                "conditions": [
                    {"name": "RSI", "op": "<", "threshold": c.strategy.rsi_oversold,
                     "current": round(rsi, 1), "met": rsi < c.strategy.rsi_oversold},
                    {"name": "Regime", "needed": "BULL", "met": is_bull},
                    {"name": "ATR", "op": ">=", "threshold": c.long_filters.atr_percentile_min,
                     "current": round(atr_pctl, 1), "met": atr_pctl >= c.long_filters.atr_percentile_min},
                    {"name": "EMA", "op": ">=", "threshold": c.long_filters.ema_separation_min,
                     "current": round(ema_sep, 2), "met": ema_sep >= c.long_filters.ema_separation_min},
                ],
            },
            "V12_SHORT": {
                "direction": "SHORT",
                "conditions": [
                    {"name": "RSI", "op": ">", "threshold": c.strategy.rsi_overbought,
                     "current": round(rsi, 1), "met": rsi > c.strategy.rsi_overbought},
                    {"name": "Regime", "needed": "BEAR", "met": not is_bull},
                ],
            },
            "BEAR_LONG": {
                "direction": "LONG",
                "conditions": [
                    {"name": "RSI", "op": "<", "threshold": c.bear_long_filters.rsi_threshold,
                     "current": round(rsi, 1), "met": rsi < c.bear_long_filters.rsi_threshold},
                    {"name": "Regime", "needed": "BEAR", "met": not is_bull},
                    {"name": "EMA", "op": ">=", "threshold": c.bear_long_filters.ema_separation_min,
                     "current": round(ema_sep, 2), "met": ema_sep >= c.bear_long_filters.ema_separation_min},
                ],
            },
            "BULL_SHORT": {
                "direction": "SHORT",
                "conditions": [
                    {"name": "RSI", "op": ">", "threshold": c.bull_short_filters.rsi_threshold,
                     "current": round(rsi, 1), "met": rsi > c.bull_short_filters.rsi_threshold},
                    {"name": "Regime", "needed": "BULL", "met": is_bull},
                    {"name": "ATR", "op": ">=", "threshold": c.bull_short_filters.atr_percentile_min,
                     "current": round(atr_pctl, 1), "met": atr_pctl >= c.bull_short_filters.atr_percentile_min},
                    {"name": "EMA", "op": ">=", "threshold": c.bull_short_filters.ema_separation_min,
                     "current": round(ema_sep, 2), "met": ema_sep >= c.bull_short_filters.ema_separation_min},
                ],
            },
        }
