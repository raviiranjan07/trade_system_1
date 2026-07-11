"""THE training entry point — one command trains any registered model.

    PYTHONPATH=src python -m training.train --model <model_key>

Structure:
  - architectures.py   the model zoo (nn.Modules by registry name)
  - training_utils.py  the shared ritual (data, splits, scaler, fit,
                       export, manifest, registration)
  - this file          MODEL_SPECS (paths/names per model) + TASKS
                       (targets + loss + eval per learning task)

A model = a params.yaml block (architecture, task, features, training,
inference, split) + a MODEL_SPECS row. A new model with an existing
task shape = config only. A new task = one Task function here.

The retired 2026 H1 tasks (binary_direction_h96, binary_direction_
mfe_aux_h8, exit_aware_3class) live in git history — they are complete
worked examples of the ritual below.

Every task function follows the same ritual (all helpers in
training_utils / feature_lib — never copy formulas):

    def run_my_task(model_key, cfg, spec, out_dir, max_epochs, register):
        ranges = tu.split_ranges(cfg["split"])          # date-based splits
        tu.set_seeds(cfg["training"]["seed"])
        fc, lb = tu.load_cache_and_labels(spec["labels"])
        X_raw = feature_lib.<formulas>(...)              # ONE source of truth
        lb, dates, X_raw = tu.align_to_labels(fc, lb, X_raw)
        splits = tu.split_by_dates(dates, valid_mask, ranges)
        mean, std = tu.fit_scaler(X_raw, splits["train"])  # train-only fit
        model = architectures.build(cfg["architecture"], **arch_kwargs)
        model, best_vl, epochs = tu.fit(model, loader, train_loss, val_loss,
                                        optimizer, ...)
        <evaluate train/val/test — protocol decides required metrics>
        tu.export_model(out_dir, model, ..., scaler_arrays={...})  # ONNX + pt
        if register:
            with run_experiment(experiment_name=spec["experiment"],
                                protocol_name=spec["protocol"], ...) as run:
                run.log_metrics(...)                     # protocol-enforced
            tu.register_staging(spec["model_name"], run_id, source)
        tu.write_manifest(out_dir, ...)                  # birth certificate

Smoke-test flags (used by verification, never by the pipeline):
  --max-epochs N   override params.yaml epochs
  --out-dir PATH   write artifacts elsewhere (default: models/<X>_staging)
  --no-register    skip MLflow logging + registry (artifacts still written)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine.signals import feature_lib  # noqa: F401  (tasks import formulas from here)
from training import architectures  # noqa: F401
from training import training_utils as tu

logger = logging.getLogger(__name__)

REPO_ROOT = tu.REPO_ROOT
FEATURES_DIR = REPO_ROOT / "data/features/direction_prediction"

# One row per trainable model. `protocol` names the configs/protocols/
# YAML this model's runs are validated against (per-model — never
# hardcode a protocol name inside a task function).
MODEL_SPECS: dict[str, dict] = {
    # "my_model": {
    #     "model_name": "MY_MODEL",                    # registry + models/ dir name
    #     "out_dir": "models/MY_MODEL_staging",
    #     "labels": FEATURES_DIR / "exit_aware_labels.parquet",
    #     "experiment": "my_experiment",               # MLflow experiment
    #     "protocol": "my_protocol_v1",                # configs/protocols/<name>.yaml
    #     "default_task": "my_task",
    #     "default_architecture": "my_encoder",
    # },
}

# One entry per learning task (label scheme + loss + eval). See the
# ritual in the module docstring; full worked examples in git history.
TASKS: dict[str, callable] = {
    # "my_task": run_my_task,
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not MODEL_SPECS:
        sys.exit("No models registered in MODEL_SPECS (training/train.py). "
                 "Add the new architecture's spec row, task, and params.yaml "
                 "block first — see the module docstring for the ritual.")

    ap = argparse.ArgumentParser(description="Unified model training entry point")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override params.yaml max_epochs (smoke tests)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override artifact dir (smoke tests)")
    ap.add_argument("--no-register", action="store_true",
                    help="skip MLflow logging + registry (smoke tests)")
    args = ap.parse_args()

    spec = MODEL_SPECS[args.model]
    cfg = tu.load_params(args.model)
    task_name = cfg.get("task", spec["default_task"])
    if task_name not in TASKS:
        raise KeyError(f"Unknown task {task_name!r}. Available: {sorted(TASKS)}")
    out_dir = args.out_dir or (REPO_ROOT / spec["out_dir"])

    print(f"=== training.train | model={args.model} task={task_name} "
          f"arch={cfg.get('architecture', spec['default_architecture'])} ===")
    TASKS[task_name](args.model, cfg, spec, out_dir, args.max_epochs,
                     register=not args.no_register)


if __name__ == "__main__":
    main()
