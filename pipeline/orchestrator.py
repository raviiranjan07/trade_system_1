"""
Unified Pipeline Orchestrator

Runs all trading system stages in sequence:
1. State Vector Generation
2. Regime Labeling
3. Outcome Labeling
4. Similarity Analysis
5. Decision Generation
"""

import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from exceptions import (
    DatabaseConnectionError,
    ConfigurationError,
    DataValidationError,
    MissingDataError,
    PipelineStageError,
)
from data.raw.ohlcv_loader import OHLCVLoader
from data.validators.data_integrity import validate_ohlcv
from decision.decision_engine import DecisionEngine
from features.location import compute_location_features
from features.momentum import compute_momentum_features
from features.trend import compute_trend_features
from features.volatility import compute_volatility_features
from features.volume import compute_volume_features
from outcomes.outcome_labeler import label_outcomes
from regime.regime_labeler import label_regime_row, smooth_regime
from similarity.similarity_engine import SimilarityEngine
from state.normalizer import RollingNormalizer
from state.state_builder import build_state
from state.state_store import save_state_vectors_parquet


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""

    success: bool
    stages_completed: list
    stages_failed: list
    state_df: Optional[pd.DataFrame] = None
    regime_df: Optional[pd.DataFrame] = None
    outcome_df: Optional[pd.DataFrame] = None
    similarity_result: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: float = 0.0


class PipelineOrchestrator:
    """
    Orchestrates the complete trading pipeline.

    Stages:
        1. state_vectors: Fetch data, compute features, build state vectors
        2. regime_labeling: Classify market regimes
        3. outcome_labeling: Compute MFE/MAE outcomes
        4. similarity: Find similar historical states
        5. decision: Generate trading decision

    Usage:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()

        # Or run specific stages
        result = orchestrator.run(stages=["state_vectors", "regime_labeling"])
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to config YAML file. Uses default if None.
        """
        self.config = get_config(config_path)
        self.logger = self._setup_logging()

        # Pipeline state
        self.state_df: Optional[pd.DataFrame] = None
        self.regime_df: Optional[pd.DataFrame] = None
        self.outcome_df: Optional[pd.DataFrame] = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging based on config settings."""
        logger = logging.getLogger("pipeline")
        logger.setLevel(getattr(logging, self.config.get("logging.level", "INFO")))

        # Clear existing handlers
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if self.config.get("logging.console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if self.config.get("logging.file_output", True):
            logs_dir = Path(self.config.get("paths.logs_dir", "logs"))
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_file = logs_dir / self.config.get("logging.log_file", "pipeline.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        base_dir = Path(self.config.get("paths.data_dir", "data"))

        for subdir in ["state_vectors_dir", "regimes_dir", "outcomes_dir"]:
            dir_path = base_dir / self.config.get(f"paths.{subdir}", subdir.replace("_dir", ""))
            dir_path.mkdir(parents=True, exist_ok=True)

        logs_dir = Path(self.config.get("paths.logs_dir", "logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        stages: Optional[list] = None,
        pair: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the complete pipeline or specific stages.

        Args:
            stages: List of stages to run. Runs all if None.
            pair: Override pair from config
            start_date: Override start date from config
            end_date: Override end date from config

        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()
        self._ensure_directories()

        # Get parameters
        pair = pair or self.config.get("data.pair")
        timeframe = self.config.get("data.timeframe", "1m")
        start_date = start_date or self.config.get("data.start_date")
        end_date = end_date or self.config.get("data.end_date")

        # Determine which stages to run
        if stages is None:
            pipeline_config = self.config.get_section("pipeline").get("stages", {})
            stages = [s for s, enabled in pipeline_config.items() if enabled]

        all_stages = ["state_vectors", "regime_labeling", "outcome_labeling", "similarity", "decision"]
        stages = [s for s in all_stages if s in stages]

        self.logger.info("=" * 60)
        self.logger.info(f"PIPELINE START | {pair} | {start_date} to {end_date}")
        self.logger.info(f"Stages to run: {stages}")
        self.logger.info("=" * 60)

        completed = []
        failed = []
        result = PipelineResult(
            success=True,
            stages_completed=[],
            stages_failed=[],
        )

        try:
            # Stage 1: State Vectors
            if "state_vectors" in stages:
                self.logger.info("[1/5] Building state vectors...")
                self.state_df = self._run_state_vectors(pair, start_date, end_date, timeframe)
                completed.append("state_vectors")
                result.state_df = self.state_df
                self.logger.info(f"      State vectors built: {len(self.state_df)} rows")
            else:
                # Load existing state vectors
                self._load_state_vectors(pair, timeframe)

            # Stage 2: Regime Labeling
            if "regime_labeling" in stages:
                self.logger.info("[2/5] Labeling regimes...")
                self.regime_df = self._run_regime_labeling(pair, timeframe)
                completed.append("regime_labeling")
                result.regime_df = self.regime_df
                self.logger.info(f"      Regime distribution:\n{self.regime_df['regime'].value_counts().to_string()}")
            else:
                self._load_regimes(pair, timeframe)

            # Stage 3: Outcome Labeling
            if "outcome_labeling" in stages:
                self.logger.info("[3/5] Labeling outcomes...")
                self.outcome_df = self._run_outcome_labeling(pair, timeframe)
                completed.append("outcome_labeling")
                result.outcome_df = self.outcome_df
                self.logger.info(f"      Outcomes computed for {len(self.outcome_df)} states")
                # Show outcome statistics
                self._log_outcome_stats(self.outcome_df)
            else:
                self._load_outcomes(pair, timeframe)

            # Stage 4: Similarity Analysis
            if "similarity" in stages:
                self.logger.info("[4/5] Running similarity analysis...")
                result.similarity_result = self._run_similarity(pair, timeframe)
                completed.append("similarity")
                self.logger.info(f"      Similarity result: expectancy={result.similarity_result.get('expectancy', 'N/A'):.4f}")

            # Stage 5: Decision
            if "decision" in stages:
                self.logger.info("[5/5] Generating decision...")
                result.decision = self._run_decision(pair, timeframe)
                completed.append("decision")
                self.logger.info(f"      Decision: {result.decision.get('action', 'N/A')} {result.decision.get('direction', '')}")

        except DatabaseConnectionError as e:
            current_stage = stages[len(completed)] if len(completed) < len(stages) else "unknown"
            self.logger.error(f"DATABASE CONNECTION ERROR in stage '{current_stage}'")
            self.logger.error(str(e))
            self.logger.error("")
            self.logger.error("Troubleshooting:")
            self.logger.error("  1. Check if PostgreSQL is running")
            self.logger.error("  2. Verify DATABASE_URL in .env or config/config.yaml")
            self.logger.error("  3. Ensure the database exists and is accessible")
            result.success = False
            result.error = str(e)
            failed.append(current_stage)

        except ConfigurationError as e:
            current_stage = stages[len(completed)] if len(completed) < len(stages) else "unknown"
            self.logger.error(f"CONFIGURATION ERROR in stage '{current_stage}'")
            self.logger.error(str(e))
            result.success = False
            result.error = str(e)
            failed.append(current_stage)

        except DataValidationError as e:
            current_stage = stages[len(completed)] if len(completed) < len(stages) else "unknown"
            self.logger.error(f"DATA VALIDATION ERROR in stage '{current_stage}'")
            self.logger.error(str(e))
            result.success = False
            result.error = str(e)
            failed.append(current_stage)

        except MissingDataError as e:
            current_stage = stages[len(completed)] if len(completed) < len(stages) else "unknown"
            self.logger.error(f"MISSING DATA ERROR in stage '{current_stage}'")
            self.logger.error(str(e))
            result.success = False
            result.error = str(e)
            failed.append(current_stage)

        except FileNotFoundError as e:
            current_stage = stages[len(completed)] if len(completed) < len(stages) else "unknown"
            self.logger.error(f"FILE NOT FOUND in stage '{current_stage}'")
            self.logger.error(f"Missing file: {e}")
            self.logger.error("")
            self.logger.error("Suggestion: Run earlier pipeline stages first to generate required files.")
            result.success = False
            result.error = str(e)
            failed.append(current_stage)

        except Exception as e:
            current_stage = stages[len(completed)] if len(completed) < len(stages) else "unknown"
            self.logger.error(f"UNEXPECTED ERROR in stage '{current_stage}'")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.debug(f"Stack trace:\n{traceback.format_exc()}")
            result.success = False
            result.error = str(e)
            failed.append(current_stage)

        # Finalize result
        execution_time = (datetime.now() - start_time).total_seconds()
        result.stages_completed = completed
        result.stages_failed = failed
        result.execution_time_seconds = execution_time

        self.logger.info("=" * 60)
        self.logger.info(f"PIPELINE {'COMPLETE' if result.success else 'FAILED'}")
        self.logger.info(f"Completed: {completed}")
        if failed:
            self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Execution time: {execution_time:.2f}s")
        self.logger.info("=" * 60)

        return result

    def _run_state_vectors(
        self, pair: str, start_date: str, end_date: str, timeframe: str
    ) -> pd.DataFrame:
        """Stage 1: Build state vectors from raw data."""
        # Fetch from DB
        loader = OHLCVLoader()
        df = loader.fetch_ohlcv(pair=pair, start_time=start_date, end_time=end_date)

        # Validate with gap handling from config
        max_gap = self.config.get("data.max_gap_tolerance", 0)
        fill_gaps = self.config.get("data.fill_small_gaps", False)
        df = validate_ohlcv(df, max_gap_tolerance=max_gap, fill_gaps=fill_gaps)

        # Compute features
        df = compute_trend_features(df)
        df = compute_momentum_features(df)
        df = compute_volatility_features(df)
        df = compute_volume_features(df)
        df = compute_location_features(df)

        # Normalize
        window = self.config.get("normalization.window", 2000)
        norm = RollingNormalizer(window)

        df["ema50_slope_z"] = norm.zscore(df["ema50_slope"])
        df["ema200_slope_z"] = norm.zscore(df["ema200_slope"])
        df["return_5m_z"] = norm.zscore(df["return_5m"])
        df["return_15m_z"] = norm.zscore(df["return_15m"])
        df["rsi_z"] = norm.zscore(df["rsi_14"])
        df["volume_z"] = norm.zscore(df["volume_raw"])
        df["vwap_distance_z"] = norm.zscore(df["vwap_distance"])
        df["atr_percentile"] = norm.percentile(df["atr_14"])

        # Build states
        state_df = df.dropna().copy()
        state_df["state"] = state_df.apply(build_state, axis=1)

        # Save
        save_state_vectors_parquet(df=state_df, pair=pair, timeframe=timeframe)

        return state_df

    def _run_regime_labeling(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Stage 2: Label market regimes."""
        if self.state_df is None:
            self._load_state_vectors(pair, timeframe)

        df = self.state_df.copy()

        # Label regimes
        df["regime_raw"] = df.apply(label_regime_row, axis=1)

        # Smooth
        window = self.config.get("regime.smoothing_window", 30)
        df["regime"] = smooth_regime(df["regime_raw"], window=window)

        # Save
        base_dir = Path(self.config.get("paths.data_dir", "data"))
        regimes_dir = base_dir / self.config.get("paths.regimes_dir", "regimes")
        output_path = regimes_dir / f"{pair}_{timeframe}_regimes.parquet"
        df[["regime"]].to_parquet(output_path, engine="pyarrow")

        return df[["regime"]]

    def _run_outcome_labeling(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Stage 3: Compute outcome labels (MFE/MAE)."""
        if self.state_df is None:
            self._load_state_vectors(pair, timeframe)

        # Fetch prices
        loader = OHLCVLoader()
        ohlcv = loader.fetch_ohlcv(pair=pair)
        close_prices = ohlcv["close"].loc[self.state_df.index]

        # Label outcomes
        outcome_df = label_outcomes(
            state_df=self.state_df,
            price_series=close_prices,
            pair=pair,
            timeframe=timeframe,
        )

        return outcome_df

    def _run_similarity(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Stage 4: Find similar historical states."""
        if self.outcome_df is None:
            self._load_outcomes(pair, timeframe)
        if self.regime_df is None:
            self._load_regimes(pair, timeframe)

        # Initialize engine
        k = self.config.get("similarity.k", 200)
        engine = SimilarityEngine(
            outcome_df=self.outcome_df,
            regime_df=self.regime_df,
            k=k,
        )

        # Query with latest state
        current_state = self.outcome_df.iloc[-1]
        current_regime = self.regime_df.iloc[-1]["regime"]
        horizon = self.config.get("similarity.default_horizon", 30)

        result = engine.query(
            current_state=current_state,
            regime=current_regime,
            horizon=horizon,
        )

        return result

    def _run_decision(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Stage 5: Generate trading decision."""
        if self.outcome_df is None:
            self._load_outcomes(pair, timeframe)
        if self.regime_df is None:
            self._load_regimes(pair, timeframe)

        # Initialize engines
        k = self.config.get("similarity.k", 200)
        similarity_engine = SimilarityEngine(
            outcome_df=self.outcome_df,
            regime_df=self.regime_df,
            k=k,
        )

        decision_engine = DecisionEngine(
            capital=self.config.get("decision.capital", 10000),
            risk_per_trade=self.config.get("decision.risk_per_trade", 0.005),
        )

        # Get current context
        current_state = self.outcome_df.iloc[-1]
        current_regime = self.regime_df.iloc[-1]["regime"]
        horizon = self.config.get("similarity.default_horizon", 30)

        # Query similarity
        sim_result = similarity_engine.query(
            current_state=current_state,
            regime=current_regime,
            horizon=horizon,
        )

        # Make decision
        decision = decision_engine.decide(
            similarity_result=sim_result,
            regime=current_regime,
        )

        return decision

    def _load_state_vectors(self, pair: str, timeframe: str) -> None:
        """Load existing state vectors from disk."""
        base_dir = Path(self.config.get("paths.data_dir", "data"))
        states_dir = base_dir / self.config.get("paths.state_vectors_dir", "state_vectors")
        path = states_dir / f"{pair}_{timeframe}_state.parquet"

        if not path.exists():
            raise FileNotFoundError(f"State vectors not found: {path}")

        self.state_df = pd.read_parquet(path)

    def _load_regimes(self, pair: str, timeframe: str) -> None:
        """Load existing regime labels from disk."""
        base_dir = Path(self.config.get("paths.data_dir", "data"))
        regimes_dir = base_dir / self.config.get("paths.regimes_dir", "regimes")
        path = regimes_dir / f"{pair}_{timeframe}_regimes.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Regime labels not found: {path}")

        self.regime_df = pd.read_parquet(path)

    def _load_outcomes(self, pair: str, timeframe: str) -> None:
        """Load existing outcome labels from disk."""
        base_dir = Path(self.config.get("paths.data_dir", "data"))
        outcomes_dir = base_dir / self.config.get("paths.outcomes_dir", "outcomes")
        path = outcomes_dir / f"{pair}_{timeframe}_outcomes.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Outcome labels not found: {path}")

        self.outcome_df = pd.read_parquet(path)

    def _log_outcome_stats(self, outcome_df: pd.DataFrame) -> None:
        """Log outcome statistics after labeling."""
        horizons = [10, 30, 120]

        self.logger.info("")
        self.logger.info("      Outcome Statistics (LONG positions):")
        self.logger.info("      " + "-" * 50)
        self.logger.info(f"      {'Horizon':<12} {'Avg MFE':>12} {'Avg MAE':>12} {'Expectancy':>12}")
        self.logger.info("      " + "-" * 50)

        for h in horizons:
            mfe_col = f"mfe_long_{h}m"
            mae_col = f"mae_long_{h}m"

            if mfe_col in outcome_df.columns and mae_col in outcome_df.columns:
                avg_mfe = outcome_df[mfe_col].mean() * 100
                avg_mae = outcome_df[mae_col].mean() * 100
                expectancy = avg_mfe + avg_mae  # MAE is negative

                self.logger.info(
                    f"      {h:>3}m         {avg_mfe:>+11.3f}% {avg_mae:>+11.3f}% {expectancy:>+11.3f}%"
                )

        self.logger.info("      " + "-" * 50)
        self.logger.info("")
