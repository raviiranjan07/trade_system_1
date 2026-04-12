"""Log Colab GPU attention results into MLOps (registry + MLflow)."""

import csv
from pathlib import Path
from mlops import tracking
from mlops.registry import RegistryRow, append_run

tracking.init()

# Remove wrong local runs (0.35 threshold, CPU)
reg_path = Path("experiments/mlops_registry.csv")
rows = []
with open(reg_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if "LSTMAttention" in row.get("model_type", "") or row.get("model_type", "") == "LSTMConnected":
            continue
        rows.append(row)

with open(reg_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Removed local wrong-threshold runs. Kept {len(rows)} rows.")

# Colab GPU results
colab_runs = [
    {
        "run_name": "LSTM_baseline_colab_gpu",
        "model_type": "LSTMConnected",
        "temperature": "N/A",
        "metrics": {
            "test_accuracy": 0.514, "test_confident_accuracy": 0.575,
            "test_n_confident": 1481, "test_n_confident_long": 895, "test_n_confident_short": 586,
            "test_confident_accuracy_long": 0.564, "test_confident_accuracy_short": 0.590,
            "test_long_short_ratio": 1.5,
            "test_avg_mfe_long": 192.0, "test_avg_mfe_short": 209.0,
            "test_avg_mae_long": 210.0, "test_avg_mae_short": 172.0,
            "backtest_n_trades": 1428, "backtest_win_rate": 40.0,
            "backtest_total_bps": 2984.0, "backtest_profit_factor": 1.17,
            "backtest_long_bps": 4589.0, "backtest_short_bps": -1606.0,
        },
        "primary": "0.575",
        "notes": "LSTM baseline (Config A). Colab T4 GPU, 0.40 SHORT threshold. +2984 bps, PF 1.17.",
    },
    {
        "run_name": "Attn_temp1.0_colab_gpu",
        "model_type": "LSTMAttention_temp1.0",
        "temperature": "1.0",
        "metrics": {
            "test_accuracy": 0.515, "test_confident_accuracy": 0.579,
            "test_n_confident": 1194, "test_n_confident_long": 556, "test_n_confident_short": 638,
            "test_confident_accuracy_long": 0.581, "test_confident_accuracy_short": 0.577,
            "test_long_short_ratio": 0.9,
            "test_avg_mfe_long": 208.0, "test_avg_mfe_short": 217.0,
            "test_avg_mae_long": 223.0, "test_avg_mae_short": 173.0,
            "backtest_n_trades": 1191, "backtest_win_rate": 42.8,
            "backtest_total_bps": 3204.0, "backtest_profit_factor": 1.22,
            "backtest_long_bps": 5553.0, "backtest_short_bps": -2350.0,
        },
        "primary": "0.579",
        "notes": "Attention temp=1.0. Colab T4 GPU, 0.40 threshold. +3204 bps, PF 1.22.",
    },
    {
        "run_name": "Attn_temp0.5_colab_gpu_BEST",
        "model_type": "LSTMAttention_temp0.5",
        "temperature": "0.5",
        "metrics": {
            "test_accuracy": 0.514, "test_confident_accuracy": 0.594,
            "test_n_confident": 834, "test_n_confident_long": 510, "test_n_confident_short": 324,
            "test_confident_accuracy_long": 0.602, "test_confident_accuracy_short": 0.580,
            "test_long_short_ratio": 1.6,
            "test_avg_mfe_long": 220.0, "test_avg_mfe_short": 206.0,
            "test_avg_mae_long": 212.0, "test_avg_mae_short": 173.0,
            "backtest_n_trades": 920, "backtest_win_rate": 47.7,
            "backtest_total_bps": 7271.0, "backtest_profit_factor": 1.69,
            "backtest_long_bps": 7777.0, "backtest_short_bps": -506.0,
        },
        "primary": "0.594",
        "notes": "BEST: Attention temp=0.5. Colab T4 GPU, 0.40 threshold. +7271 bps, PF 1.69. Beats MLP production.",
    },
    {
        "run_name": "Attn_temp0.1_colab_gpu",
        "model_type": "LSTMAttention_temp0.1",
        "temperature": "0.1",
        "metrics": {
            "test_accuracy": 0.513, "test_confident_accuracy": 0.578,
            "test_n_confident": 1399, "test_n_confident_long": 595, "test_n_confident_short": 804,
            "test_confident_accuracy_long": 0.578, "test_confident_accuracy_short": 0.577,
            "test_long_short_ratio": 0.7,
            "test_avg_mfe_long": 206.0, "test_avg_mfe_short": 223.0,
            "test_avg_mae_long": 225.0, "test_avg_mae_short": 187.0,
            "backtest_n_trades": 1355, "backtest_win_rate": 42.2,
            "backtest_total_bps": 3094.0, "backtest_profit_factor": 1.18,
            "backtest_long_bps": 4092.0, "backtest_short_bps": -998.0,
        },
        "primary": "0.578",
        "notes": "Attention temp=0.1. Colab T4 GPU, 0.40 threshold. +3094 bps, PF 1.18.",
    },
    {
        "run_name": "Attn_temp0.05_colab_gpu",
        "model_type": "LSTMAttention_temp0.05",
        "temperature": "0.05",
        "metrics": {
            "test_accuracy": 0.515, "test_confident_accuracy": 0.581,
            "test_n_confident": 1552, "test_n_confident_long": 882, "test_n_confident_short": 670,
            "test_confident_accuracy_long": 0.575, "test_confident_accuracy_short": 0.590,
            "test_long_short_ratio": 1.3,
            "test_avg_mfe_long": 206.0, "test_avg_mfe_short": 223.0,
            "test_avg_mae_long": 216.0, "test_avg_mae_short": 193.0,
            "backtest_n_trades": 1479, "backtest_win_rate": 42.8,
            "backtest_total_bps": 6453.0, "backtest_profit_factor": 1.36,
            "backtest_long_bps": 8535.0, "backtest_short_bps": -2082.0,
        },
        "primary": "0.581",
        "notes": "Attention temp=0.05. Colab T4 GPU, 0.40 threshold. +6453 bps, PF 1.36.",
    },
]

for r in colab_runs:
    run_id = tracking.start_run("direction_prediction", run_name=r["run_name"])
    tracking.log_params({
        "model_type": r["model_type"],
        "features": "roc_diff+rsi_diff+rp_diff+sma200_diff x 8 = 32",
        "label": "direction_h8",
        "horizon": "8",
        "threshold_long": "0.60",
        "threshold_short": "0.40",
        "hardware": "Colab_T4_GPU",
        "temperature": r["temperature"],
    })
    tracking.log_metrics(r["metrics"])
    tracking.set_tags({
        "source": "colab_eval_attention_v2.py",
        "notes": r["notes"],
    })
    tracking.end_run()

    append_run(RegistryRow(
        run_id=f"colab_{r['model_type']}",
        experiment_name="direction_prediction",
        protocol_name="direction_prediction_v1",
        status="FINISHED",
        start_time="2026-04-13",
        user="raviranjan",
        source_name="scripts/colab_eval_attention_v2.py",
        model_type=r["model_type"],
        dataset_version="feature_cache_cleaned_23col+labels_h8",
        primary_metric_name="test_confident_accuracy",
        primary_metric_value=r["primary"],
        notes=r["notes"],
    ))
    print(f"  Logged: {r['run_name']}")

print(f"\nDone. 5 Colab runs logged to MLflow + registry.")
