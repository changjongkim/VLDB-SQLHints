# HALO: Hardware-Aware Learned Optimizer (v3)
> **A unified SQL hint recommendation framework that adapts to quantitative hardware differences**

---

## Overview

HALO is a framework for recommending SQL optimizer hints that are **safe to apply across different hardware environments**. It solves a critical problem in database management: hints tuned on one server can cause severe performance regressions when deployed on a server with different storage (NVMe vs SATA) or CPU (Intel vs AMD) characteristics.

**Version 3 Enhancements:**
- **Quantitative HW Awareness:** Moves beyond binary flags (0/1) to measurable specs (IOPS, MB/s, Clock GHz).
- **Workload Metadata Enrichment:** Incorporates table sizes, row counts, and index metadata into risk prediction.
- **Improved σ (Sigma) Model:** Achieves $R^2 \approx 0.60$ and **85%+ Direction Accuracy** on cross-environment transfers.

---

## Key Results (v3 retrained)

```
  Metric                               HALO-G       HALO-R (v3)
  =======================================================
  Total Regressions (12 transfers)         13            4
  Regression Reduction Rate                 —          69.2%
  Total Rescues (G failed, R saved)         —            9
  σ Model R² (Determination)           -0.31         0.604
  σ Model Ranking Accuracy               59%           82.7%
  =======================================================
```

---

## Usage Guide

### 1. Target Environment Specification

HALO v3 requires quantitative specs of the target server. You can register them in three ways:

#### A. Manual Registration
```python
from hw_registry import register_env

register_env('MyServer', {
    'server': 'Xeon', 'storage': 'NVMe', 'cpu': 'Xeon-4310',
    'seq_read_mbps': 3500, 'rand_read_iops': 500000,
    'cpu_cores': 12, 'cpu_boost_ghz': 3.3, 'ram_gb': 64, 'buffer_pool_gb': 20
})
```

#### B. Automatic Local Discovery
For running directly on the target machine:
```python
from hw_registry import auto_register_local_env

# Automatically detects CPU/RAM and applies representative I/O specs for NVMe
auto_register_local_env(env_id='Target_Server', storage_type='NVMe')
```

#### C. Command Line Recommendation
```bash
python3 src/generate_hinted_queries.py
```
*(Currently configured to generate TPC-H/JOB recommendations for EPYC Target Servers)*

---

### 2. Integration Example (API)

```python
from halo_framework import HaloFramework

halo = HaloFramework()
# Data enrichment automatically happens during training (v3)
halo.train() 

# Predict if a specific plan is safe for a new target
res = halo.recommend_for_new_query(
    source_env='A_NVMe', 
    target_env='B_SATA', 
    plan_str=my_explain_analyze_json
)
print(res)
```

---

## Project Structure (v3)

```
halo/
├── src/
│   ├── hw_registry.py           # ★ Single source of truth for HW specs
│   ├── workload_metadata.py     # ★ DB Table metadata (Size, Rows, Indexes)
│   ├── sigma_model_v3.py        # Retrained σ model with 79 quantitative features
│   ├── halo_framework.py        # Core logic (Robust filtering & Recommendation)
│   ├── generate_hinted_queries.py # Batch generator for production SQL
│   ├── evaluate_halo.py         # Full cross-environment comparison script
│   └── visualize.py             # Dashboard & Plot generation
├── data/
│   ├── unified_queries.parquet   # Execution logs
│   └── unified_operators.parquet # Operator-level metrics
└── results/
    ├── sigma_model_v3.pkl       # Trained model binary
    └── figures/                 # Evaluation plots (HPP, Speedup, Heatmaps)
```

---

## The σ (Sigma) Model v3 — Quantitative Features

The model uses **79 features** to predict hardware sensitivity:
- **HW Ratios:** `log(Target_IOPS / Source_IOPS)`, `cpu_clock_ratio`, etc.
- **Workload:** `table_rows`, `table_size_mb`, `fits_in_buffer`.
- **Cross-Terms (Critical):** `scan_x_iops`, `join_x_cpu_clock`, `table_size_x_storage_speed`.

---

## Running the Evaluation

```bash
# 1. Update/Retrain the model
python3 src/sigma_model_v3.py

# 2. Run all generalization experiments (E1-E8)
python3 src/experiments.py

# 3. View cross-environment head-to-head results
python3 src/evaluate_halo.py

# 4. Generate all figures
python3 src/visualize.py
```

---

## Citation
```
HALO v3: Quantitative Hardware-Aware Learned Optimizer 
Big Data Lab Research
```
