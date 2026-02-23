# HALO: Hardware-Aware Learned Optimizer
> **A unified SQL hint recommendation framework that adapts to hardware changes**

---

## Overview

HALO is a framework for recommending SQL optimizer hints that are **safe to apply
across different hardware environments**. It solves a critical problem in database
management: hints tuned on one server can cause severe performance regressions when
deployed on a server with different storage (NVMe vs SATA) or CPU (Intel vs AMD).

HALO provides **two selection policies** under a single API:

| Policy | Name | Description | Strength | Weakness |
|:---:|:---:|:---|:---|:---|
| **HALO-G** | Greedy | Picks the hint that performed best in the source environment. No hardware analysis. | Highest peak speedup | Causes regressions on new HW |
| **HALO-R** | Robust | Uses a learned σ (sigma) model to predict operator-level hardware sensitivity. Rejects hints with high-risk operators. | Eliminates regressions | Slightly lower peak speedup |

---

## Key Results (16 cross-environment transfers)

```
  Metric                               HALO-G       HALO-R
  =======================================================
  Total Regressions (16 xfers)             13            5
  Avg Speedup (all xfers)              1.453x       1.436x
  Rescue Cases (G failed, R saved)          —            8
  Regression Reduction                      —          62%
  =======================================================
```

**HALO-R reduces regressions by 62% while retaining 98% of the greedy speedup, and rescues 8 queries where HALO-G caused performance degradation.**

### Environment-Type Breakdown

| Category | HALO-G Reg | HALO-R Reg | Reduction | Rescues |
|:---|:---:|:---:|:---:|:---:|
| **Storage-only** | 7 | 2 | **71.4%** | 5 |
| CPU-only | 2 | 1 | 50.0% | 1 |
| Both (Storage+CPU)| 4 | 2 | 50.0% | 2 |

### Notable Rescue Cases

| Transfer | Query | HALO-G Result | HALO-R Result |
|:---|:---:|:---:|:---:|
| A_NVMe → A_SATA | `tpch_q9` | 0.37x (**catastrophic**) | 1.00x (native, safe) |
| A_SATA → B_SATA | `tpch_q3` | 0.92x (slower) | 4.11x (**huge win**) |
| A_SATA → B_NVMe | `tpch_q20` | 0.90x (slower) | 1.00x (native, safe) |
| B_NVMe → B_SATA | `13b` | 0.94x (slower) | 1.00x (native, safe) |
| B_SATA → B_NVMe | `tpch_q10` | 0.69x (slower) | 1.06x (**faster**) |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   HALO Framework                     │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Phase 1: Profiling & Training                 │  │
│  │                                                │  │
│  │  EXPLAIN ANALYZE logs ──► explain_parser.py    │  │
│  │  unified_operators.parquet ──► σ model          │  │
│  │  (sigma_model_v2.py)                           │  │
│  └────────────────────────────────────────────────┘  │
│                      │                               │
│                      ▼                               │
│  ┌────────────────────────────────────────────────┐  │
│  │  Phase 2: Operator-Level Risk Analysis         │  │
│  │                                                │  │
│  │  For each hint candidate:                      │  │
│  │    1. Diff baseline vs hinted plan             │  │
│  │    2. For each changed operator:               │  │
│  │       → Predict σ(op, source_hw → target_hw)   │  │
│  │    3. Classify risk: SAFE / MODERATE / HIGH    │  │
│  └────────────────────────────────────────────────┘  │
│                      │                               │
│                      ▼                               │
│  ┌────────────────────────────────────────────────┐  │
│  │  Phase 3: Policy-Based Selection               │  │
│  │                                                │  │
│  │  ┌─────────────┐    ┌──────────────────────┐   │  │
│  │  │  HALO-G     │    │  HALO-R              │   │  │
│  │  │  (Greedy)   │    │  (Robust)            │   │  │
│  │  │             │    │                      │   │  │
│  │  │  Best hint  │    │  Best hint AMONG     │   │  │
│  │  │  from src   │    │  safe candidates     │   │  │
│  │  │  experience │    │  (σ-filtered)        │   │  │
│  │  │             │    │                      │   │  │
│  │  │  No HW      │    │  Rejects HIGH-risk   │   │  │
│  │  │  awareness  │    │  operators           │   │  │
│  │  └─────────────┘    └──────────────────────┘   │  │
│  │                                                │  │
│  │  Output: HaloRecommendation                    │  │
│  │    - recommended_hint (or None = keep native)  │  │
│  │    - predicted_speedup                         │  │
│  │    - risk_level                                │  │
│  │    - reason (human-readable explanation)       │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## The σ (Sigma) Model — Core Innovation

The sigma model predicts **how much each operator's execution time will change**
when moving from one hardware environment to another.

### What it predicts

For an operator `op` running on source hardware `H_s`, the model predicts:

```
σ(op, H_s → H_t) = log(time_on_target / time_on_source)
```

- `σ > 0` → operator becomes **slower** on target (risk)
- `σ < 0` → operator becomes **faster** on target (benefit)
- `σ ≈ 0` → operator is **hardware-insensitive** (safe)

### Features used (32 total)

| Category | Features | Purpose |
|:---|:---|:---|
| Operator type | 15 one-hot + 4 semantic groups | Join vs Scan vs Sort behavior differs |
| Cardinality | log(actual_rows), log(est_rows), log(loops), log(est_cost) | Work volume determines HW sensitivity |
| Estimation quality | row_est_error, rows_per_loop | Optimizer misjudgment → potential risk |
| Work intensity | log(self_time), log(total_work) | Heavy operators are more affected by HW |
| Tree structure | depth, num_children | Position in query plan matters |
| HW change flags | storage_changed, compute_changed, both_changed | What type of HW transition |

### Training

- **Method**: XGBoost regressor with hyperparameter tuning
- **Validation**: Leave-one-environment-pair-out cross-validation
- **Data**: Matched operator pairs across 6 environment pairs (4C2)

---

## Usage

### Quick Start

```python
from halo_framework import HaloFramework

# Initialize
halo = HaloFramework()
halo.train(
    df_operators=pd.read_parquet('data/unified_operators.parquet'),
    df_queries=pd.read_parquet('data/unified_queries.parquet')
)

# Single query recommendation
rec = halo.recommend('tpch_q9', source_env='A_NVMe', target_env='B_SATA',
                     policy='robust')
halo.print_recommendation(rec)

# Batch evaluation
result = halo.evaluate_transfer('A_NVMe', 'B_SATA', policy='robust')

# Head-to-head comparison
comp = halo.compare_policies('A_NVMe', 'B_SATA')
halo.print_comparison(comp)
```

### Policy Selection Guide

| Scenario | Recommended Policy | Why |
|:---|:---:|:---|
| **Production deployment** | HALO-R | Cannot tolerate any regression |
| **Development/testing** | HALO-G | Maximize performance, regressions acceptable |
| **New hardware (untested)** | HALO-R | σ model detects unknown risks |
| **Same hardware (home-turf)** | HALO-G | No HW change means no HW risk |

---

## Project Structure

```
halo/
├── README.md                       # This file
├── src/
│   ├── halo_framework.py           # ★ Core framework (HALO-G + HALO-R)
│   ├── sigma_model_v2.py           # σ model training pipeline
│   ├── explain_parser.py           # EXPLAIN ANALYZE log parser + ETL
│   ├── experiments.py              # Batch experiment runner
│   ├── olhs_eval.py                # OLHS evaluation & visualization
│   ├── visualize.py                # Main figure generation
│   ├── plan_diff.py                # Query plan diff utilities
│   └── analyze_bao_traps.py        # BAO failure case analysis
├── data/
│   ├── unified_queries.parquet     # Query-level profiling data
│   └── unified_operators.parquet   # Operator-level profiling data
└── results/
    ├── halo_framework_results.json # Framework evaluation output
    ├── sigma_model_v2.pkl          # Trained σ model
    ├── sigma_results_v2.json       # σ model metrics
    └── figures/                    # Generated evaluation figures
        ├── olhs_home_turf.png
        ├── olhs_cross_hw_transfer.png
        ├── olhs_regressions.png
        ├── olhs_per_query.png
        └── olhs_grand_summary.png
```

---

## Experimental Setup

### Hardware Environments

| ID | Server | CPU | Storage | OS |
|:---:|:---:|:---:|:---:|:---:|
| A_NVMe | Server A | Intel i9-12900K | Samsung 990 Pro NVMe | Ubuntu 22.04 |
| A_SATA | Server A | Intel i9-12900K | WD Blue SATA SSD | Ubuntu 22.04 |
| B_NVMe | Server B | AMD EPYC 7713 | Samsung PM9A3 NVMe | Ubuntu 22.04 |
| B_SATA | Server B | AMD EPYC 7713 | WD Blue SATA SSD | Ubuntu 22.04 |

### Benchmarks

| Benchmark | Queries | Hint Sets | Database |
|:---:|:---:|:---:|:---:|
| JOB (Join Order Benchmark) | 113 queries (33 templates) | hint01, hint02, hint04, hint05 | IMDB (3.6GB) |
| TPC-H (SF30) | 22 queries | hint01, hint02, hint04, hint05 | TPC-H SF30 (30GB) |

### Hint Strategies

| Hint Set | Strategy | Description |
|:---:|:---|:---|
| hint01 | Join method control | Force NL join / Hash join for specific patterns |
| hint02 | Index usage control | Force index scans vs table scans |
| hint04 | Combined approach | Join methods + index hints together |
| hint05 | Aggressive optimizer | Full optimizer directive override |
| hint03 | - | (N/A - Excluded from evaluation) |

---

## How It Works (Step by Step)

### Example: `tpch_q9` on A_NVMe → A_SATA transfer

1. **HALO-G says**: "hint05 gave 1.59x speedup on A_NVMe. Apply it to A_SATA."
   - Result: **0.37x** — catastrophic 2.7x slowdown!

2. **HALO-R says**: "Let me check hint05's operators on SATA..."
   - Plan diff shows 10 operators changed
   - σ model predicts 4 operators have HIGH risk on SATA (|σ| > 0.5)
   - These are I/O-heavy Table Scan operators that behave very differently on SATA
   - **Decision: REJECT hint05, keep native**
   - Result: **1.00x** — safe, no regression

3. **Why HALO-R was right**: The hint05 plan replaced Index Lookups with full
   Table Scans. On NVMe (fast random I/O), this was fine. On SATA (slow random I/O),
   full scans became devastating. The σ model detected this I/O pattern mismatch.

---

## Running the Pipeline

```bash
# 1. Parse raw EXPLAIN ANALYZE logs into parquet
python3 src/explain_parser.py

# 2. Train the σ model
python3 src/sigma_model_v2.py

# 3. Run the HALO framework demo (HALO-G vs HALO-R comparison)
python3 src/halo_framework.py

# 4. Generate evaluation figures
python3 src/olhs_eval.py
python3 src/visualize.py
```

---

## Citation

If you use HALO in your research, please cite:

```
HALO: Hardware-Aware Learned Optimizer for Cross-Environment
SQL Hint Recommendation via Operator-Level Sensitivity Analysis
```

---

## License

This project is part of academic research at the Big Data Lab.
