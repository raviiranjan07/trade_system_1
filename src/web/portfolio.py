"""Binance Futures Portfolio Data Client.

Authenticated client for reading account data from Binance USD-M Futures.
Read-only — no order placement, only account/position/order/income queries.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PortfolioClient:
    """Async Binance Futures client for portfolio data."""

    def __init__(self):
        self._client = None
        self._api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self._api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_secret)

    async def connect(self) -> bool:
        if not self.is_configured:
            return False
        try:
            from binance import AsyncClient
            self._client = await AsyncClient.create(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
            return True
        except Exception as e:
            logger.error("Binance auth failed: %s", e)
            return False

    async def close(self):
        if self._client:
            try:
                await self._client.close_connection()
            except Exception:
                pass
            self._client = None

    async def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """Fetch full portfolio data from Binance Futures."""
        if not self._client:
            ok = await self.connect()
            if not ok:
                return {"error": "not_configured"}

        try:
            account, balances, positions, orders, income = await asyncio.gather(
                self._client.futures_account(),
                self._client.futures_account_balance(),
                self._client.futures_position_information(),
                self._client.futures_get_all_orders(symbol="BTCUSDT", limit=50),
                self._client.futures_income_history(limit=100),
            )
        except Exception as e:
            logger.error("Portfolio fetch error: %s", e)
            # Properly close then reset so next call retries connection
            await self.close()
            return {"error": str(e)}

        # Account summary
        summary = {
            "total_wallet_balance": float(account.get("totalWalletBalance", 0)),
            "total_unrealized_pnl": float(account.get("totalUnrealizedProfit", 0)),
            "total_margin_balance": float(account.get("totalMarginBalance", 0)),
            "available_balance": float(account.get("availableBalance", 0)),
            "total_position_margin": float(account.get("totalPositionInitialMargin", 0)),
        }

        # Non-zero balances
        parsed_balances = [
            {
                "asset": b["asset"],
                "balance": float(b["balance"]),
                "available": float(b.get("availableBalance", b.get("withdrawAvailable", 0))),
                "cross_pnl": float(b.get("crossUnPnl", 0)),
            }
            for b in balances
            if abs(float(b.get("balance", 0))) > 0.0001
        ]

        # Non-zero positions
        parsed_positions = [
            {
                "symbol": p.get("symbol", ""),
                "side": "LONG" if float(p.get("positionAmt", 0)) > 0 else "SHORT",
                "size": abs(float(p.get("positionAmt", 0))),
                "entry_price": float(p.get("entryPrice", 0)),
                "mark_price": float(p.get("markPrice", 0)),
                "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
                "leverage": int(p.get("leverage", 1)),
                "margin_type": p.get("marginType", ""),
            }
            for p in positions
            if abs(float(p.get("positionAmt", 0))) > 0.0001
        ]

        # Recent orders (newest first)
        parsed_orders = sorted(
            [
                {
                    "order_id": o.get("orderId", 0),
                    "symbol": o.get("symbol", ""),
                    "side": o.get("side", ""),
                    "type": o.get("type", ""),
                    "status": o.get("status", ""),
                    "price": float(o.get("price", 0)),
                    "qty": float(o.get("origQty", 0)),
                    "filled_qty": float(o.get("executedQty", 0)),
                    "time": o.get("time", 0),
                }
                for o in orders
            ],
            key=lambda x: x["time"],
            reverse=True,
        )

        # Income history (realized PNL, funding, commission)
        parsed_income = sorted(
            [
                {
                    "type": i.get("incomeType", ""),
                    "amount": float(i.get("income", 0)),
                    "asset": i.get("asset", ""),
                    "symbol": i.get("symbol", ""),
                    "time": i.get("time", 0),
                }
                for i in income
            ],
            key=lambda x: x["time"],
            reverse=True,
        )

        return {
            "summary": summary,
            "balances": parsed_balances,
            "positions": parsed_positions,
            "orders": parsed_orders,
            "income": parsed_income,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_pnl_summary(self, period: str) -> Dict[str, Any]:
        """Fetch income history for a time period and compute PNL breakdown.

        Args:
            period: 'week', 'month', 'year', or 'all'
        """
        if not self._client:
            ok = await self.connect()
            if not ok:
                return {"error": "not_configured"}

        now = datetime.now(timezone.utc)
        # Binance only keeps last 3 months of income data.
        # Without startTime it returns only 7 days.
        # Always pass startTime, capped at 90 days for year/all.
        if period == "week":
            start = now - timedelta(days=7)
        elif period == "month":
            start = now - timedelta(days=30)
        else:  # year or all — Binance caps at ~90 days
            start = now - timedelta(days=90)

        start_ms = int(start.timestamp() * 1000)

        try:
            # Paginate — Binance returns max 1000 per call
            all_income: List[dict] = []
            kwargs = {"limit": 1000, "startTime": start_ms}
            while True:
                batch = await self._client.futures_income_history(**kwargs)
                if not batch:
                    break
                all_income.extend(batch)
                if len(batch) < 1000:
                    break
                # Move cursor past last item
                kwargs["startTime"] = int(batch[-1].get("time", 0)) + 1
        except Exception as e:
            logger.error("PNL fetch error: %s", e)
            await self.close()
            return {"error": str(e)}

        # Aggregate by type
        realized = 0.0
        funding = 0.0
        commission = 0.0
        other = 0.0
        for i in all_income:
            amt = float(i.get("income", 0))
            t = i.get("incomeType", "")
            if t == "REALIZED_PNL":
                realized += amt
            elif t == "FUNDING_FEE":
                funding += amt
            elif t == "COMMISSION":
                commission += amt
            else:
                other += amt

        net = realized + funding + commission + other
        return {
            "period": period,
            "realized": round(realized, 4),
            "funding": round(funding, 4),
            "commission": round(commission, 4),
            "other": round(other, 4),
            "net": round(net, 4),
            "trade_count": len([i for i in all_income if i.get("incomeType") == "REALIZED_PNL"]),
            "total_entries": len(all_income),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
