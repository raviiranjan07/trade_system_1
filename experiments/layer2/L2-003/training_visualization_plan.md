# Training Visualization Plan — ML V2 (LSTM + Attention)

**Date:** 2026-04-15
**Status:** Plan (not implemented)
**Owner:** TBD

---

## Goal

Build a live, interactive visualization that lets us **watch the ML V2 model train epoch-by-epoch** and see:

1. **What it's learning** (training set behavior)
2. **How it's applying that learning** (validation set behavior)
3. **What internal patterns are forming** (attention, hidden state, gate activations)

Not just "is loss going down?" — but **what kind of model is emerging?**

---

## Why this matters

- **Catch overfitting early.** Watch the train/val gap widen and stop before the model starts memorizing.
- **Build trust.** See whether the model learns real patterns or random correlations.
- **Tune intuition.** Know which epoch produced the most generalizable model — not just lowest val loss.
- **Debug strange trades.** When live model takes a weird trade, we can replay what it "learned" up to that checkpoint.

---

## What we want to see, per epoch

### Layer 1 — Headline metrics (the basics)
- Train loss + val loss curves
- Train accuracy + val accuracy
- Learning rate schedule
- Confidence-filtered val accuracy (>0.60 / <0.40)

### Layer 2 — Behavior on samples (the meat)
- **Prediction distribution** — histogram of P(LONG) over val set. At epoch 1 it's a tight cluster at 0.5. By epoch 20 it spreads bimodally if the model is learning.
- **Confusion matrix** evolving — how is it splitting LONG vs SHORT?
- **Per-sample tracking** — pick 50 fixed val samples, plot how their predictions change over epochs. Shows whether the model is converging on a stable view or flip-flopping.

### Layer 3 — Internal state (the why)
- **Attention weights** averaged across val set — heatmap (epoch × bar position). Watch which bars the model decides to focus on as training progresses.
- **Hidden state activation patterns** — PCA-projected to 2D, colored by direction. Watch clusters form.
- **Gate activations** (forget/input/output) — average sigmoid output per gate per epoch. Shows when the model "decides" to start using memory vs ignoring it.

### Layer 4 — Diagnostic
- Gradient norms (per layer) — exploding? vanishing?
- Weight norms — are weights growing unbounded?
- Effective learning rate (after schedule)
- Time per epoch + ETA

---

## What it'd look like (live dashboard)

```
┌─ Training Dashboard — ML V2 (Attention temp=0.5) ──────────┐
│                                                            │
│  Epoch 23/100  ●●●●●●●●●●●●●●●●●●●●●●●○○○○○○○○○○○○○○○○○○  │
│                                                            │
│  Train: loss 0.534 ↓  acc 59.1% ↑                          │
│  Val:   loss 0.587 ↓  acc 56.8% ↑                          │
│  Best epoch so far: 21 (val_loss=0.582)                    │
│                                                            │
│  ┌─ Loss curves ──────────────────────────────────┐        │
│  │      ╲                                         │        │
│  │       ╲                                        │        │
│  │        ╲___________train                       │        │
│  │            ╲___________val                     │        │
│  │                                                │        │
│  └────────────────────────────────────────────────┘        │
│                                                            │
│  ┌─ Attention weights (val avg) ──────────────────┐        │
│  │  -8 ▎ -7 ▍ -6 ▌ -5 ▋ -4 ███ -3 █████ -2 ████  │        │
│  │  -1 ██▏                                        │        │
│  │  ▲ Stabilized: model focuses on bars -2, -3   │        │
│  └────────────────────────────────────────────────┘        │
│                                                            │
│  ┌─ Prediction distribution (val) ────────────────┐        │
│  │  P(LONG):  ▁▂▄▆▇█▇▆▄▂▁                         │        │
│  │            0.0  0.2  0.4  0.6  0.8  1.0       │        │
│  │  ▲ Bimodal — confident in both directions     │        │
│  └────────────────────────────────────────────────┘        │
│                                                            │
│  ┌─ Confusion matrix (val) ───────────────────────┐        │
│  │             Pred LONG    Pred SHORT             │        │
│  │  Actual UP    420 ✓        180 ✗               │        │
│  │  Actual DN    140 ✗        440 ✓               │        │
│  └────────────────────────────────────────────────┘        │
│                                                            │
│  ┌─ Per-sample tracker (50 fixed val samples) ───┐        │
│  │  Heatmap: epoch × sample_id                   │        │
│  │  Color: P(LONG) — 0=red, 1=green             │        │
│  │  Shows convergence/flip-flopping              │        │
│  └────────────────────────────────────────────────┘        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Implementation approach

### Tooling decision

| Option | Pros | Cons |
|---|---|---|
| **TensorBoard** | Free, native to PyTorch, works in Colab | Limited custom widgets |
| **Weights & Biases (wandb)** | Beautiful UI, custom plots, free for personal | Sends data to cloud |
| **MLflow (existing)** | Already set up | Less interactive than wandb |
| **Custom matplotlib + IPython.display** | Full control, no external deps | Build everything ourselves |

**Recommendation: Start with wandb.** Easiest path to a rich live dashboard. Falls back to MLflow for offline use.

### Where to plug it in

The training loop in `scripts/colab/colab_eval_attention_v2.py:198-227` (the `train_model` function). Add hooks at:

- **Per epoch end:** log loss, acc, learning rate
- **Per N epochs:** run validation, log distributions, attention, confusion matrix
- **Per N epochs:** compute internal state stats (gradient norms, weight norms)

### Code structure

```
scripts/colab/
├── colab_eval_attention_v2.py        (existing — main script)
├── viz/
│   ├── __init__.py
│   ├── metrics_logger.py            (loss/acc/LR per epoch)
│   ├── prediction_logger.py         (per-sample tracking, distributions)
│   ├── attention_logger.py          (attention weights, hidden state)
│   └── diagnostic_logger.py         (gradients, weights)
└── train_with_viz.py                (entry point — wraps train_model)
```

### Modifications to model

Update `LSTMAttention.forward()` to optionally return:
```python
def forward(self, x, return_internals=False):
    all_h, _ = self.lstm(x)
    scores = self.attn_score(all_h).squeeze(-1)
    attn_weights = torch.softmax(scores / self.temperature, dim=1)
    attended = torch.bmm(attn_weights.unsqueeze(1), all_h).squeeze(1)
    # ... rest of forward
    if return_internals:
        return p_mu, p_md, p_dir, {
            "attn_weights": attn_weights,
            "all_hidden_states": all_h,
            "attended": attended,
        }
    return p_mu, p_md, p_dir
```

### Per-sample tracking

Pre-select 50 validation samples (stratified: 25 LONG-truth, 25 SHORT-truth). After each epoch:
1. Run those 50 samples through the model
2. Save predictions to a `[epoch, sample_id]` matrix
3. End of training → render as heatmap (color = P(LONG))

This shows whether the model is **converging on a view** or constantly **flip-flopping** on the same inputs.

---

## Phased rollout

### Phase 1 — Headline metrics (quick win, ~2 hours)
- Add wandb integration to `train_model`
- Log: train_loss, val_loss, train_acc, val_acc, learning_rate per epoch
- Get a basic loss curve dashboard going

**Deliverable:** Run training in Colab, watch loss/acc on wandb in real-time.

### Phase 2 — Behavior on samples (~3 hours)
- Compute val predictions every epoch
- Log:
  - Prediction distribution histogram
  - Confusion matrix
  - Confidence-filtered accuracy
  - Per-sample tracking matrix (50 fixed samples)

**Deliverable:** Heatmap of how predictions on the same samples evolve over epochs.

### Phase 3 — Internal state (~4 hours)
- Modify model to expose attention + hidden states
- Log:
  - Average attention weights (val set) per epoch
  - PCA of attended states colored by direction
  - Gate activation averages (forget, input, output)

**Deliverable:** "Attention focus heatmap" showing which bars the model attends to over training.

### Phase 4 — Diagnostic (~2 hours)
- Hook into backward pass
- Log gradient norms per layer
- Log weight norms
- Log effective learning rate

**Deliverable:** Catch exploding/vanishing gradients early.

**Total: ~11 hours of work** for full implementation across all 4 phases.

---

## Open questions (need decision before starting)

1. **wandb or MLflow?** wandb is more interactive, MLflow keeps everything local. The user has MLflow already set up (`mlruns/` exists).
2. **All 5 configs (A-E) or just one model?** Logging 5 simultaneous training runs is wandb's strength — easy to compare. But adds complexity.
3. **Training speed impact?** Per-epoch validation adds ~5-10% overhead. Acceptable for a dashboard.
4. **Per-sample tracking — which 50 samples?** Stratified random? Manually-curated edge cases? Both?
5. **Do we want to keep the dashboard data after training?** wandb stores forever (free tier limits apply). Useful for retro analysis.

---

## What we'd LEARN from this

After running training with full visualization:

| Question | How the dashboard answers it |
|---|---|
| Is the model actually learning? | Train loss going down, val loss following |
| When does it start overfitting? | Train acc keeps rising, val acc plateaus |
| What patterns does it focus on? | Attention weights stabilize on specific bars |
| Is it confident or hesitant? | Prediction distribution: bimodal = confident, peaked at 0.5 = uncertain |
| Which epoch is best? | Lowest val_loss + reasonable train/val gap |
| Is training stable? | No exploding gradients, no flip-flopping predictions |
| Does early stopping make sense? | Per-sample tracker shows when predictions stabilize |

---

## Risk / what could go wrong

- **wandb rate limits** — the free tier has limits. Logging too frequently could throttle.
- **Colab disconnects** — long training with wandb logging is fine, but if Colab disconnects, training restarts from scratch unless we add checkpointing.
- **Visualization overhead** — per-epoch validation + attention compute could slow training 10-20%.
- **Over-interpretation** — humans see meaning in attention patterns even when they're meaningless. Need to be disciplined about distinguishing real patterns from noise.

---

## Next steps

1. **Discuss the open questions above** — decide on tooling and scope.
2. **Start with Phase 1** (~2 hours) to prove the pipeline works.
3. **Iterate based on what's most useful** — may not need all 4 phases.

---

## Reference: where the training code lives

- **Main training script:** [scripts/colab/colab_eval_attention_v2.py](scripts/colab/colab_eval_attention_v2.py)
  - `train_model()` function: lines 198-227
  - `LSTMAttention` class: lines 154-176
  - Training loop runs all 5 configs (A-E) sequentially
- **Loss function:** combined MSE (MFE up + MFE down) + 5× BCE (direction)
- **Optimizer:** Adam, LR=0.001, weight_decay=0.0
- **Scheduler:** ReduceLROnPlateau, patience=5, factor=0.5
- **Early stopping:** patience=10
- **Hidden size:** 128
- **Dropout:** 0.5
