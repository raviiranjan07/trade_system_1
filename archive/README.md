# Archive

Retired artifacts kept for reference (not used by any code or pipeline).

- `model_cards/` — cards of the retired 2026 H1 model lineage
  (ML_V1 MLP, ML_V3 exit-aware LSTM+Attention, and never-deployed
  lstm_attention / lstm_gru / mlp_curriculum research architectures).
  The models themselves were deleted in the 2026-07-11 clean slate;
  weights are recoverable only from git history (pre-wipe commits).
- `data_cards/` — cards of retired dataset artifacts (old direction
  labels + old feature cache — both deleted; their producer stages were
  removed from dvc.yaml).
- `dvc_3model_pipeline.yaml` — the full 18-stage DVC pipeline that
  trained/verified/backtested the 3-model lineage. The live dvc.yaml
  was reduced to the shared skeleton stages; use this as reference
  when writing the new architecture's stage chain.
