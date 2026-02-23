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
  Total Regressions (16 xfers)             13            8
  Avg Speedup (all xfers)              1.453x       1.438x
  Rescue Cases (G failed, R saved)          —            5
  Regression Reduction                      —          38%
  =======================================================
```

**HALO-R (v1.3) achieves a 38% reduction in regressions while maintaining 99% of the greedy speedup, providing a near-optimal balance between performance and stability.**

### Environment-Type Breakdown

| Category | HALO-G Reg | HALO-R Reg | Reduction | Significant Wins |
|:---|:---:|:---:|:---:|:---:|
| **Storage-only** | 7 | 4 | 43% | Avoided `tpch_q9` crash |
| CPU-only | 2 | 1 | 50% | Safer Join selection |
| Both (Storage+CPU)| 4 | 3 | 25% | Balanced Cross-Xfer |

### Notable Rescue Cases

| Transfer | Query | HALO-G Result | HALO-R Result |
|:---|:---:|:---:|:---:|
| A_NVMe → A_SATA | `tpch_q9` | 0.37x (**catastrophic**) | 1.00x (native, safe) |
| A_SATA → B_SATA | `tpch_q3` | 0.92x (slower) | 4.11x (**huge win**) |
| A_SATA → B_NVMe | `tpch_q20` | 0.90x (slower) | 1.00x (native, safe) |
| B_NVMe → B_SATA | `13b` | 0.94x (slower) | 1.00x (native, safe) |
| B_SATA → B_NVMe | `tpch_q10` | 0.69x (slower) | 1.06x (**faster**) |

### Top 10 Cross-Environment Accelerations (TPC-H)
| Rank | Transfer | Query | Hint | Predicted Trend (Source) | Actual Target Speedup |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | B_SATA → B_NVMe | `tpch_q2` | `hint02` | 3.35x | **4.41x** |
| 2 | A_SATA → A_NVMe | `tpch_q3` | `hint05` | 3.38x | **4.15x** |
| 3 | B_SATA → B_NVMe | `tpch_q17` | `hint02` | 2.58x | **2.91x** |
| 4 | A_SATA → A_NVMe | `tpch_q13` | `hint05` | 2.81x | **2.88x** |
| 5 | A_NVMe → A_SATA | `tpch_q11` | `hint01` | 1.98x | **2.81x** |
| 6 | A_SATA → A_NVMe | `tpch_q8` | `hint05` | 2.17x | **2.39x** |
| 7 | A_SATA → A_NVMe | `tpch_q5` | `hint04` | 1.63x | **2.16x** |
| 8 | B_SATA → B_NVMe | `tpch_q12` | `hint02` | 1.71x | **1.90x** |
| 9 | A_NVMe → A_SATA | `tpch_q7` | `hint05` | 1.45x | **1.55x** |
| 10 | A_NVMe → A_SATA | `tpch_q21` | `hint01` | 1.30x | **1.52x** |

### Top 10 Cross-Environment Accelerations (JOB)
| Rank | Transfer | Query | Hint | Predicted Trend (Source) | Actual Target Speedup |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | B_SATA → B_NVMe | `15b` | `hint05` | 1.09x | **282.03x** |
| 2 | B_SATA → B_NVMe | `11b` | `hint01` | 2.50x | **166.96x** |
| 3 | B_SATA → B_NVMe | `6c` | `hint01` | 1.29x | **55.45x** |
| 4 | B_SATA → B_NVMe | `11a` | `hint01` | 1.13x | **52.92x** |
| 5 | B_SATA → B_NVMe | `13c` | `hint01` | 1.12x | **14.97x** |
| 6 | B_SATA → B_NVMe | `18b` | `hint01` | 1.15x | **14.01x** |
| 7 | B_SATA → B_NVMe | `15a` | `hint05` | 1.08x | **12.44x** |
| 8 | B_SATA → B_NVMe | `7b` | `hint01` | 1.13x | **7.07x** |
| 9 | B_SATA → B_NVMe | `12a` | `hint01` | 1.29x | **5.19x** |
| 10 | B_SATA → B_NVMe | `19a` | `hint04` | 2.75x | **4.01x** |

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
