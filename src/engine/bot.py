"""ALPHA trading bot — thin entrypoint.

Wires everything together and runs it. All logic lives elsewhere:
  track.py            — one strategy's lifecycle (4 instances)
  orchestrator.py     — shared world: market feed, indicators, turn-taking
  persistence.py      — trades CSVs + risk-state files
  dashboard_bridge.py — TrackEvents -> dashboard updates

Run: PYTHONPATH=src python -m engine.bot
"""

import asyncio
import logging

from .config.loader import load_config
from .orchestrator import Orchestrator, build_tracks
from .risk.preflight import run_preflight

logger = logging.getLogger(__name__)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    tracks = build_tracks(config)
    orchestrator = Orchestrator(config, tracks)

    logger.info("Running risk calculator preflight...")
    if not run_preflight(wallet=tracks[0].wallet, btc_price=100000):
        logger.error("Risk calculator preflight FAILED — refusing to start")
        return
    logger.info("Preflight PASSED | wallets: %s",
                {t.name: round(t.wallet, 2) for t in tracks})

    # Dashboard web server (PYTHONPATH=src puts `web` on the path)
    from web.server import run_server
    from web.state import dashboard_state

    port = 8080
    orchestrator.attach_dashboard(dashboard_state)

    web_task = asyncio.create_task(run_server(port=port, state=dashboard_state))
    logger.info("Dashboard running at http://localhost:%d", port)

    try:
        await orchestrator.start()
    finally:
        web_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
