"""Retroactive cleanup: register all past experiments into MLOps system.

Reads from old registry, experiment folders, and PLAN/RESULTS docs.
Creates entries in mlops_registry.csv + MLflow for every past experiment.

Run once:
  PYTHONPATH=src python scripts/retroactive_cleanup.py
"""

from mlops import tracking
from mlops.registry import RegistryRow, append_run

# All past experiments, extracted from registry.csv, REGISTRY.md, folder notes, PLAN.md
EXPERIMENTS = [
    # =========================================================================
    # LAYER 0: Strategy Development (EXP-001 → SETUP-V1 → EXP-010)
    # =========================================================================
    RegistryRow(
        run_id="retro_EXP-001",
        experiment_name="EXP-001_rsi_behavior_analysis",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="analysis",
        artifacts_dir="experiments/layer1/rsi/EXP-001",
        primary_metric_name="win_rate",
        primary_metric_value="99.7",
        notes="RSI oversold/overbought bounce rate ~99%. Tight stops kill 70-86% of winners.",
    ),
    RegistryRow(
        run_id="retro_EXP-002",
        experiment_name="EXP-002_rsi_ma_trend_filter",
        status="REJECTED",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="analysis",
        artifacts_dir="experiments/layer1/rsi/EXP-002",
        primary_metric_name="win_rate",
        primary_metric_value="99.7",
        notes="REJECTED. MA filter removes 97% signals for 0.3% gain. RSI works WITH and AGAINST trend.",
    ),
    RegistryRow(
        run_id="retro_EXP-003",
        experiment_name="EXP-003_rsi_failure_deep_dive",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="analysis",
        artifacts_dir="experiments/layer1/rsi/EXP-003",
        primary_metric_name="win_rate",
        primary_metric_value="98.3",
        notes="Failures from weak signals (low volume/momentum). Filtering loses 15-40 good signals per failure.",
    ),
    RegistryRow(
        run_id="retro_EXP-004",
        experiment_name="EXP-004_sma200_regime_filter",
        status="VALIDATED",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-004",
        primary_metric_name="n_trades",
        primary_metric_value="266",
        notes="VALIDATED. SMA200 filter: LONG in bull + SHORT in bear = 266 signals, 1 failure, 0 account killers.",
    ),
    RegistryRow(
        run_id="retro_EXP-005",
        experiment_name="EXP-005_trailing_stop_optimization",
        status="VALIDATED",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-005",
        primary_metric_name="avg_winner_bps",
        primary_metric_value="18.2",
        notes="VALIDATED. LONG 20bps TS=+18.2 gross (+10.2 net). SHORT 30bps TS=+23.2 gross (+15.2 net).",
    ),
    RegistryRow(
        run_id="retro_SETUP-V1",
        experiment_name="SETUP-V1_trading_setup_locked",
        status="LOCKED",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi",
        primary_metric_name="profit_factor",
        primary_metric_value="1.97",
        notes="LOCKED. RSI<20+Bull OR RSI>80+Bear. TS 20/30bps. 266 trades, 53.8% win, +3024 bps.",
    ),
    RegistryRow(
        run_id="retro_EXP-006",
        experiment_name="EXP-006_v12_long_when_filters",
        status="VALIDATED",
        start_time="2026-02-05",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-006",
        primary_metric_name="profit_factor",
        primary_metric_value="2.47",
        notes="VALIDATED. ATR>=25 + EMA>=0.5% for LONG. V1.2: 202 trades, 56.9% win, +3250 bps, PF 2.47.",
    ),
    RegistryRow(
        run_id="retro_EXP-007",
        experiment_name="EXP-007_v13_short_when_filters",
        status="REJECTED",
        start_time="2026-02-05",
        user="raviranjan",
        model_type="backtest",
        notes="REJECTED. All 7 SHORT filter combos reduce total profit. SHORT is robust in ALL conditions.",
    ),
    RegistryRow(
        run_id="retro_EXP-008",
        experiment_name="EXP-008_range_support_strategy",
        status="FAILED",
        start_time="2026-02-05",
        user="raviranjan",
        model_type="backtest",
        primary_metric_name="profit_factor",
        primary_metric_value="1.27",
        notes="FAILED. 774 trades, PF 1.27 but -118 bps in 2025. SHORT loses badly. V1.2 far superior.",
    ),
    RegistryRow(
        run_id="retro_EXP-009",
        experiment_name="EXP-009_remove_time_exit",
        status="REJECTED",
        start_time="2026-02-05",
        user="raviranjan",
        model_type="backtest",
        primary_metric_name="profit_factor",
        primary_metric_value="1.63",
        notes="REJECTED. Removing TIME_EXIT worsens results. +3250→+2177 bps. TIME_EXIT protects portfolio.",
    ),
    RegistryRow(
        run_id="retro_EXP-010",
        experiment_name="EXP-010_ema_crossover_strategy",
        status="FAILED",
        start_time="2026-02-05",
        user="raviranjan",
        model_type="backtest",
        primary_metric_name="profit_factor",
        primary_metric_value="1.23",
        notes="FAILED. 7 EMA pairs, all ~50/50. High volume but PF 1.13-1.47. Degrades to 1.04-1.16 in 2025.",
    ),

    # =========================================================================
    # LAYER 0: Signal Expansion (EXP-011 → EXP-014)
    # =========================================================================
    RegistryRow(
        run_id="retro_EXP-011",
        experiment_name="EXP-011_four_new_strategy_screen",
        status="COMPLETE",
        start_time="2026-02-07",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-011",
        primary_metric_name="profit_factor",
        primary_metric_value="1.43",
        notes="Volume Spike best: 2243 trades, PF 1.43, +13562 bps. Vol Squeeze/Candlestick/MACD also tested.",
    ),
    RegistryRow(
        run_id="retro_EXP-012",
        experiment_name="EXP-012_v12_plus_volume_spike_reentry",
        status="COMPLETE",
        start_time="2026-02-07",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-012",
        primary_metric_name="profit_factor",
        primary_metric_value="1.99",
        notes="Config A+RE chosen for bot: 289 trades, +4182 bps, PF 1.99. Confluence PF 6.07 but only 44 trades.",
    ),
    RegistryRow(
        run_id="retro_EXP-013",
        experiment_name="EXP-013_counter_trend_bear_long_bull_short",
        status="VALIDATED",
        start_time="2026-02-09",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-013",
        primary_metric_name="profit_factor",
        primary_metric_value="3.00",
        notes="VALIDATED → V1.3. Bear LONG (RSI<10+EMA>=1.0): 71.4% win, PF 3.00. Bull SHORT: 40% win, PF 2.16.",
    ),
    RegistryRow(
        run_id="retro_EXP-014",
        experiment_name="EXP-014_level_based_counter_trend_reentry",
        status="VALIDATED",
        start_time="2026-02-07",
        user="raviranjan",
        model_type="backtest",
        artifacts_dir="experiments/layer1/rsi/EXP-014",
        primary_metric_name="profit_factor",
        primary_metric_value="3.46",
        notes="VALIDATED → V1.3.2. Level-based re-entry: +352 bps, 9 new trades 7/9 winners. RSI saturates at extremes.",
    ),

    # =========================================================================
    # LAYER 1: Risk Management (L1-EXP-001 → L1-EXP-006)
    # =========================================================================
    RegistryRow(
        run_id="retro_L1-EXP-001",
        experiment_name="L1-EXP-001_fixed_position_baseline",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1-EXP-001",
        primary_metric_name="final_wallet",
        primary_metric_value="82",
        notes="Fixed 0.001 BTC at 125x: $10→$82, 0% ruin, linear growth. Baseline established.",
    ),
    RegistryRow(
        run_id="retro_L1-EXP-002",
        experiment_name="L1-EXP-002_position_qty_scaling",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1-EXP-002",
        primary_metric_name="optimal_step",
        primary_metric_value="2.00-2.50",
        notes="Sharp cliff $1.75→$2.00: 95%→0.1% ruin. $2.00=$4.65M median, $2.50=$909K median.",
    ),
    RegistryRow(
        run_id="retro_L1-EXP-003",
        experiment_name="L1-EXP-003_kelly_criterion_analysis",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1-EXP-003",
        primary_metric_name="quarter_kelly_step",
        primary_metric_value="1.89",
        notes="Classic Quarter-Kelly=$1.89/step matches empirical ($1.85). Sharp cliff at $1.80-$1.85.",
    ),
    RegistryRow(
        run_id="retro_L1-EXP-004",
        experiment_name="L1-EXP-004_hybrid_condition_sizing",
        status="VALIDATED",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1-EXP-004",
        primary_metric_name="oos_median",
        primary_metric_value="13800000",
        notes="VALIDATED. Size down on bad conditions (Monday LONG, low ATR/EMA): +200% vs fixed $2.00.",
    ),
    RegistryRow(
        run_id="retro_L1-EXP-005",
        experiment_name="L1-EXP-005_hybrid_kelly_adaptive",
        status="REJECTED",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1-EXP-005",
        primary_metric_name="oos_median",
        primary_metric_value="15898",
        notes="REJECTED. Adaptive Kelly much worse than fixed. Bayesian too slow, DD constraint kills recovery.",
    ),
    RegistryRow(
        run_id="retro_L1-EXP-006",
        experiment_name="L1-EXP-006_final_risk_management_system",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1-EXP-006",
        primary_metric_name="median_20x",
        primary_metric_value="40979",
        notes="COMPLETE. 8-component system: cross margin, fixed 20x/25x, safety stop. Zero safety triggers historically.",
    ),

    # =========================================================================
    # LAYER 1: Re-entry Validation (L1R-001 → L1R-007)
    # =========================================================================
    RegistryRow(
        run_id="retro_L1R-001",
        experiment_name="L1R-001_v132_ground_truth_stats",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-001",
        primary_metric_name="profit_factor",
        primary_metric_value="3.46",
        notes="V1.3.2 ground truth: 220 OOS trades, 60% win, PF 3.46, +5267 bps, Kelly 0.427.",
    ),
    RegistryRow(
        run_id="retro_L1R-002",
        experiment_name="L1R-002_dollar_per_step_sweep",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-002",
        primary_metric_name="ruin_cliff",
        primary_metric_value="1.75-2.00",
        notes="Sharp ruin cliff at $1.75-$2.00/step. Small position change = life/death.",
    ),
    RegistryRow(
        run_id="retro_L1R-003",
        experiment_name="L1R-003_signal_quality_impact",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-003",
        primary_metric_name="oos_improvement",
        primary_metric_value="340-1350%",
        notes="WEAK signals lose money OOS. Tier-specific sizing boosts results 340-1350% vs uniform.",
    ),
    RegistryRow(
        run_id="retro_L1R-004",
        experiment_name="L1R-004_edge_cases_price_wallet",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-004",
        primary_metric_name="survival_zone",
        primary_metric_value="<$43-84",
        notes="BTC price and wallet size edge cases. $100K BTC is sweet spot for position sizing.",
    ),
    RegistryRow(
        run_id="retro_L1R-005",
        experiment_name="L1R-005_safety_stop_placement",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-005",
        primary_metric_name="ruin_reduction",
        primary_metric_value="6.8→0.9%",
        notes="Safety stop at 60% liq distance cuts TRAIN ruin 6.8%→0.9%. Pure insurance.",
    ),
    RegistryRow(
        run_id="retro_L1R-006",
        experiment_name="L1R-006_integrated_risk_calculator",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-006",
        notes="Combines L1R-001 through L1R-005 into version-agnostic risk calculator.",
    ),
    RegistryRow(
        run_id="retro_L1R-007",
        experiment_name="L1R-007_final_sequential_backtest",
        status="COMPLETE",
        start_time="2026-02-04",
        user="raviranjan",
        model_type="risk_sim",
        artifacts_dir="experiments/layer1/L1R-007",
        notes="End-to-end sequential backtest with real dollar equity tracking.",
    ),

    # =========================================================================
    # LAYER 2: ML — Direction Prediction (L2-003)
    # =========================================================================
    RegistryRow(
        run_id="retro_L2-003-S1",
        experiment_name="L2-003_stage1_feature_screening_695",
        protocol_name="",
        status="COMPLETE",
        start_time="2026-02-28",
        user="raviranjan",
        model_type="analysis",
        dataset_version="feature_cache_v1",
        artifacts_dir="experiments/layer2/L2-003",
        notes="695 magnitude features ranked by GOOD/BAD separation. Natural thresholds from data.",
    ),
    RegistryRow(
        run_id="retro_L2-003-S2",
        experiment_name="L2-003_stage2_lightgbm_direction",
        protocol_name="direction_prediction_v1",
        status="FAILED",
        start_time="2026-02-28",
        user="raviranjan",
        model_type="LightGBM",
        dataset_version="feature_cache_v1",
        artifacts_dir="experiments/layer2/L2-003",
        primary_metric_name="test_accuracy",
        primary_metric_value="0.505",
        notes="FAILED. 97.9% train, 50.5% OOS. Massive overfit. Flat 11178 cols loses time order.",
    ),
    RegistryRow(
        run_id="retro_L2-003-S3",
        experiment_name="L2-003_stage3_mlp_direction_v15",
        protocol_name="direction_prediction_v1",
        status="DEPLOYED",
        start_time="2026-03-01",
        user="raviranjan",
        model_type="MLP",
        dataset_version="feature_cache_v1",
        artifacts_dir="experiments/layer2/L2-003",
        primary_metric_name="test_confident_accuracy",
        primary_metric_value="0.575",
        notes="DEPLOYED as V1.5. MLP 10→128→128→1. roc1-8+range_position+rsi7. 57-58% confident acc. +11096 bps backtest.",
    ),
    RegistryRow(
        run_id="retro_L2-003-S4",
        experiment_name="L2-003_stage4_controlled_learning_5exp",
        protocol_name="direction_prediction_v1",
        status="COMPLETE",
        start_time="2026-03-26",
        user="raviranjan",
        model_type="MLP/LSTM/Attention",
        dataset_version="feature_cache_v1",
        artifacts_dir="experiments/layer2/L2-003",
        primary_metric_name="test_confident_accuracy",
        primary_metric_value="0.588",
        notes="5 experiments (asymmetry, curriculum, encoders, attention, contrastive). Best: attention 58.8%. Ceiling confirmed.",
    ),

    # =========================================================================
    # BRAIN: S/R Bounce/Break Prediction
    # =========================================================================
    RegistryRow(
        run_id="retro_EXP-B002",
        experiment_name="EXP-B002_xgboost_sr_raw_features",
        protocol_name="sr_bounce_break_v1",
        status="FAILED",
        start_time="2026-03-01",
        user="raviranjan",
        model_type="XGBoost",
        dataset_version="datasets_v1",
        artifacts_dir="experiments/brain/EXP-B002",
        primary_metric_name="test_accuracy",
        primary_metric_value="0.507",
        notes="FAILED. 13 raw S/R features, 4-class. 75.7% train, 50.7% OOS. Massive overfit.",
    ),
    RegistryRow(
        run_id="retro_SR-S1-S6",
        experiment_name="SR_stages1-6_zone_features_various",
        protocol_name="sr_bounce_break_v1",
        status="FAILED",
        start_time="2026-03-15",
        user="raviranjan",
        model_type="MLP/LSTM/XGBoost",
        dataset_version="datasets_every_bar",
        artifacts_dir="experiments/brain/SR",
        primary_metric_name="test_accuracy",
        primary_metric_value="0.509",
        notes="FAILED. Stages 1-6: various zone features, base features, KDE. All ~50% OOS. Feature separation 0.2-8.4%.",
    ),
    RegistryRow(
        run_id="retro_SR-S8a",
        experiment_name="SR_stage8a_separate_paths_6static",
        protocol_name="sr_bounce_break_v1",
        status="COMPLETE",
        start_time="2026-04-01",
        user="raviranjan",
        model_type="MLP",
        dataset_version="datasets_every_bar",
        artifacts_dir="experiments/brain/SR",
        primary_metric_name="test_accuracy",
        primary_metric_value="0.517",
        notes="Simple separate paths (6 static). Dynamic 9→4, Static 6→4, Combine 8→2. 51.7% OOS.",
    ),
    RegistryRow(
        run_id="retro_SR-S8cB",
        experiment_name="SR_stage8cB_separate_paths_14static_mfe",
        protocol_name="sr_bounce_break_v1",
        status="COMPLETE",
        start_time="2026-04-01",
        user="raviranjan",
        model_type="MLP",
        dataset_version="datasets_every_bar",
        artifacts_dir="experiments/brain/SR",
        primary_metric_name="test_accuracy",
        primary_metric_value="0.520",
        notes="Best S/R result. 14 static + MFE features, simple separate paths. 52.0% OOS.",
    ),
    RegistryRow(
        run_id="retro_SR-S8c-hier",
        experiment_name="SR_stage8c_hierarchical_gated_fusion",
        protocol_name="sr_bounce_break_v1",
        status="COMPLETE",
        start_time="2026-04-01",
        user="raviranjan",
        model_type="MLP_hierarchical",
        dataset_version="datasets_every_bar",
        artifacts_dir="experiments/brain/SR",
        primary_metric_name="test_accuracy",
        primary_metric_value="0.516",
        notes="Hierarchical 5-step gated fusion. 174 params. Gate nearly fixed (std 0.02). 51.6% — worse than 8cB.",
    ),
]


def main():
    print("=" * 70)
    print("RETROACTIVE CLEANUP: Registering past experiments")
    print("=" * 70)

    tracking.init()

    for i, row in enumerate(EXPERIMENTS):
        # Create MLflow run
        mlflow_run_id = tracking.start_run(
            experiment_name=row.experiment_name,
            run_name=row.run_id,
        )

        # Log what we have as params
        params = {}
        if row.model_type:
            params["model_type"] = row.model_type
        if row.dataset_version:
            params["dataset_version"] = row.dataset_version
        if row.primary_metric_name:
            params["primary_metric"] = row.primary_metric_name
        if params:
            tracking.log_params(params)

        # Log primary metric if numeric
        if row.primary_metric_value:
            try:
                val = float(row.primary_metric_value)
                tracking.log_metric(row.primary_metric_name, val)
            except (ValueError, TypeError):
                pass

        # Tag as retroactive
        tracking.set_tags({
            "retroactive": "true",
            "status": row.status,
            "notes": row.notes[:250] if row.notes else "",
        })
        if row.protocol_name:
            tracking.set_tag("protocol", row.protocol_name)

        tracking.end_run(status="FINISHED")

        # Append to registry
        append_run(row)

        print(f"  [{i+1:2d}/{len(EXPERIMENTS)}] {row.run_id:<30s} {row.experiment_name}")

    print()
    print(f"Done. Registered {len(EXPERIMENTS)} experiments.")
    print(f"Registry: experiments/mlops_registry.csv")
    print(f"MLflow:   mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
