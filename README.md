# HALO: Hardware-Aware Learned Optimizer
> **A Multi-Source, Operator-Level SQL Hint Recommendation Framework That Adapts to Quantitative Hardware Differences**

---

## 1. Problem Statement

Database optimizer hints (e.g., `/*+ SET_VAR(optimizer_switch=...) */`) can dramatically improve query performance on one server, but **cause severe regressions on another**. This happens because:

- **Storage gap**: A hint optimized for NVMe (500K IOPS) may force full table scans that cripple a SATA disk (200 IOPS).
- **CPU gap**: A complex nested-loop join hint tuned on a 5.2 GHz i9 can timeout on a 2.1 GHz Xeon.
- **Memory gap**: A hash join requiring 20 GB memory works on a 128 GB server but thrashes on a 31 GB server.

**HALO solves this** by predicting, at the **individual operator level**, whether a hint will remain safe when transferred from one hardware environment to another — before ever executing it on the target.

---

## 2. Core Architecture

HALO operates as a 5-stage pipeline:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ Stage 1      │    │ Stage 2       │    │ Stage 3        │    │ Stage 4         │    │ Stage 5       │
│ Data Ingest  │───>│ Feature       │───>│ σ Model        │───>│ Risk            │───>│ SQL           │
│ & Parsing    │    │ Engineering   │    │ Training       │    │ Assessment      │    │ Generation    │
│              │    │               │    │                │    │ & Selection     │    │               │
│ EXPLAIN logs │    │ 79 features   │    │ Random Forest  │    │ HALO-R Policy   │    │ Hinted .sql   │
│ → Operator   │    │ per operator  │    │ predicts       │    │ Multi-Source    │    │ files ready   │
│   trees      │    │ pair          │    │ δ(time)        │    │ Global Ensemble │    │ to execute    │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────────┘    └──────────────┘
```

---

## 3. Stage 1 — Data Ingestion (`explain_parser.py`)

### What it does
Parses MySQL 8.0 `EXPLAIN ANALYZE` output from experiment log files and extracts a structured operator tree for every query execution.

### How it works

1. **File Discovery** (`discover_data_sources()`):
   - Scans `/root/kcj_sqlhint/` for Server A (Intel i9) and Server B (AMD EPYC) logs.
   - Supports NVMe and SATA storage variants for each server.
   - Supports both JOB and TPC-H benchmarks, including hint variants (hint01–hint05).
   
2. **Tree Parsing** (`parse_tree()`):
   - Uses regex to extract each operator node: `→ <description> (cost=X rows=Y) (actual time=A..B rows=C loops=D)`
   - Determines parent-child relationships via **indentation depth**.
   - Classifies operator types into 15 categories: `NL_INNER_JOIN`, `TABLE_SCAN`, `INDEX_LOOKUP`, `HASH_JOIN`, `AGGREGATE`, `SORT`, etc.

3. **Self-Time Computation** (`compute_self_times()`):
   ```
   self_time = (actual_time_end - actual_time_start) - Σ(children_time)
   total_work = self_time × loops
   row_est_error = log(actual_rows / estimated_rows)    # Q-error
   ```

### Output
- `unified_operators.parquet`: ~138K operator instances with 20+ fields each.
- `unified_queries.parquet`: query-level execution times per (env, query, hint_set).

---

## 4. Stage 2 — Feature Engineering (79 Features)

### Overview
Each operator is described by **79 features** organized into 5 categories, enabling the σ model to understand both *what the operator does* and *how the hardware transition affects it*.

### Feature Categories

| # | Category | Count | Key Features | Purpose |
|---|----------|-------|-------------|---------|
| 1 | **Operator Type** (one-hot + semantic) | 19 | `is_NL_INNER_JOIN`, `is_TABLE_SCAN`, `is_join`, `is_scan`, `is_index_op`, `is_agg_sort` | What kind of work this operator performs |
| 2 | **Cardinality & Work** | 13 | `log_actual_rows`, `log_est_cost`, `row_est_error`, `log_self_time`, `log_total_work`, `cost_per_row`, `time_per_row`, `selectivity` | How heavy this operator is |
| 3 | **Hardware Transition** | 16 | `iops_ratio`, `cpu_clock_ratio`, `storage_speed_ratio`, `ram_ratio`, `is_storage_downgrade`, `is_cpu_downgrade` | How different is the target hardware |
| 4 | **Workload Metadata** | 5 | `log_table_rows`, `log_table_size_mb`, `n_indexes`, `fits_in_buffer` | What data this operator touches |
| 5 | **Cross-Terms (Interaction)** | 26 | `scan_x_iops`, `join_x_cpu_clock`, `table_size_x_storage_speed`, `work_x_compute` | **Critical**: How operator type × HW change interact |

### Cross-Term Logic (The Key Innovation)
The most important features are the **interaction terms**. They capture insights like:

- `scan_x_iops = is_scan × log(target_IOPS / source_IOPS)` → "A table scan on a storage downgrade is dangerous"
- `join_x_cpu_clock = is_join × log(target_GHz / source_GHz)` → "A nested loop join on a slower CPU is risky"
- `table_size_x_storage_speed` → "A 22 GB lineitem scan is fine on NVMe but lethal on SATA"

---

## 5. Stage 3 — σ (Sigma) Model (`sigma_model_v3.py`)

### What it predicts
For a given operator running under a specific hint, the model predicts **δ(time)** — the log-ratio of execution time change when the operator is moved from the source to target hardware:

```
δ = log(self_time_target / self_time_source)
```

- δ > 0 → The operator gets **slower** on the target (risk!)
- δ < 0 → The operator gets **faster** on the target (benefit)
- δ ≈ 0 → No significant change

### Training Process

1. **Operator Matching** (`build_env_pairs_v3()`):
   - For each environment pair (e.g., `A_NVMe ↔ B_SATA`), match the **same operator** across both environments using `(query_id, hint_set, op_type, table_alias)` as the key.
   - Compute the actual δ(time) as ground truth label.
   - Filter out operators with `self_time < 0.1ms` (too noisy to measure reliably).

2. **Model Training** (`train_sigma_v3()`):
   - Algorithm: **Random Forest Regressor** (500 trees, max_depth=12)
   - 80/20 train-test split, stratified by environment pair.
   - Evaluation: R², RMSE, Direction Accuracy, Ranking Accuracy.

### Hardware Specification Registry (`hw_registry.py`)

All hardware specs are centralized in a single registry with **6 registered environments**:

| Environment | CPU | Storage | IOPS | Clock (Boost) | RAM |
|---|---|---|---|---|---|
| `A_NVMe` | i9-12900K | Samsung 990 Pro | 1,200K | 5.2 GHz | 128 GB |
| `A_SATA` | i9-12900K | Samsung 870 EVO | 98K | 5.2 GHz | 128 GB |
| `B_NVMe` | EPYC 7713 | Samsung PM9A3 | 1,000K | 3.675 GHz | 256 GB |
| `B_SATA` | EPYC 7713 | Samsung 870 EVO | 98K | 3.675 GHz | 256 GB |
| `Xeon_NVMe` | Xeon 4310 | Generic NVMe | 500K | 3.3 GHz | 31 GB |
| `Xeon_SATA` | Xeon 4310 | Generic SATA | 80K | 3.3 GHz | 31 GB |

**Transition features** are computed as `log(target_spec / source_spec)`:
- Positive value = target is better (safe direction)
- Negative value = target is worse (risk direction)

### Workload Metadata Registry (`workload_metadata.py`)

Provides table-level statistics for operator enrichment:
- **TPC-H SF30**: 8 tables (`lineitem`: 180M rows / 22 GB, down to `region`: 5 rows)
- **JOB (IMDB)**: 21 tables (`cast_info`: 36M rows / 2.3 GB, down to `role_type`: 12 rows)
- **Alias resolution**: Maps query aliases (e.g., `ci` → `cast_info`, `l1` → `lineitem`)
- **Buffer fit**: `fits_in_buffer = 1 if table_size_mb < buffer_pool_gb × 1024`

---

## 6. Stage 4 — Risk Assessment & Hint Selection

### Policy: HALO-R (Robust) with Multi-Source Global Ensemble

This is the production recommendation engine in `generate_hinted_queries.py`.

#### Step 1: Knowledge Aggregation
For each target query, HALO scans **all known server environments** for hint success experiences:

```python
for src_env in [A_NVMe, A_SATA, B_NVMe, B_SATA]:
    for hint in [hint01, hint02, hint04, hint05]:
        speedup = baseline_time / hinted_time  # Source server's experience
        if speedup >= 1.05:   # At least 5% improvement
            → Add to candidate pool
```

#### Step 2: Per-Transition Risk Verification
Each candidate is evaluated for the **specific hardware transition** (e.g., `B_NVMe → Xeon_SATA`):

```python
hw_features = compute_hw_features(src_env, target_env)
# e.g., A_NVMe → Xeon_SATA produces:
#   iops_ratio = log(80K / 1.2M) = -2.71  (massive I/O downgrade)
#   cpu_clock_ratio = log(3.3 / 5.2) = -0.45  (CPU downgrade)

for each operator in the hinted query plan:
    features = engineer_79_features(operator, hw_features)
    σ = model.predict(features)   # predicted δ(time)
    if σ > 0.8:  → mark as HIGH_RISK

risk_ratio = count(HIGH_RISK) / total_operators
```

#### Step 3: Safety Gate (HALO-R Threshold)

```
if risk_ratio <= 30%:   → SAFE candidate (pass)
if risk_ratio > 30%:    → REJECTED (too many operators predicted to slow down)
```

#### Step 4: Global Optimal Selection
From all candidates that pass the safety gate across all source servers, pick the one with the **highest source speedup**:

```python
best = max(safe_candidates, key=lambda c: c.src_speedup)
# Result: "hint02 from B_NVMe (4.41x speedup, 20% risk)" beats
#         "hint05 from A_NVMe (2.39x speedup, 20% risk)"
```

### Hint Definitions

| Hint ID | MySQL Optimizer Switch | Strategy |
|---------|----------------------|----------|
| `hint01` | `block_nested_loop=off` | Force index-based nested loop joins |
| `hint02` | `block_nested_loop=off, batched_key_access=on, mrr=on` | BKA + Multi-Range Read |
| `hint04` | `block_nested_loop=off, hash_join=on` | Force hash joins |
| `hint05` | `block_nested_loop=off` + `NO_RANGE_OPTIMIZATION` | Disable range scan on primary key |

---

## 7. Stage 5 — SQL Generation

The final output is a directory of ready-to-execute `.sql` files:

```
hinted_queries_xeon_multi_source/
├── NVMe/
│   ├── TPCH/
│   │   ├── tpch_q3_hint05.sql      ← 4.15x speedup expected
│   │   ├── tpch_q15_native.sql     ← No safe hint found
│   │   └── ...
│   └── JOB/
│       ├── 10a_hint02.sql
│       └── ...
└── SATA/
    ├── TPCH/
    └── JOB/
```

Each SQL file contains a metadata header:
```sql
-- HALO Recommended SQL
-- Query     : tpch_q2 (TPCH)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint02
-- From Src  : B_NVMe
-- Reason    : HALO-R (Multi-Source): 'hint02' selected from B_NVMe (src_speedup=4.41x, risk=20%)
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on")
           SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */
  s_acctbal, s_name, ...
```

---

## 8. HALO-R 3-Tier Confidence System (`halo_framework.py`)

The full `HaloFramework` class implements a more granular 3-tier policy for research evaluation:

| Tier | Condition | Action |
|------|-----------|--------|
| 🟢 **GREEN** | 0 high-risk operators AND predicted speedup ≥ 1.05x | **Always apply** |
| 🟡 **YELLOW** | Some risk but Expected Value > threshold | **Apply if reward justifies risk** |
| 🔴 **RED** | Too many high-risk operators | **Reject → Native fallback** |

### Risk-Reward Expected Value
```
EV = source_speedup × (1 - risk_probability)
risk_probability = total_risk_score / n_operators
```

### Adaptive Per-Operator Thresholds
I/O-bound operators (TABLE_SCAN, INDEX_SCAN) naturally have higher variance, so they receive **relaxed thresholds**. CPU-bound operators (FILTER, AGGREGATE) receive tighter ones.

---

## 9. Project Structure

```
halo/
├── src/
│   ├── explain_parser.py           # Stage 1: EXPLAIN ANALYZE → operator trees
│   ├── hw_registry.py              # HW spec registry + transition feature computation
│   ├── workload_metadata.py        # Table metadata enrichment (TPCH/JOB)
│   ├── sigma_model_v3.py           # Stage 3: σ model training (79 features, RF)
│   ├── generate_hinted_queries.py  # Stage 4-5: Multi-Source recommendation + SQL gen
│   ├── halo_framework.py           # Full HALO-G/R API with 3-tier policy
│   ├── evaluate_halo.py            # Cross-environment evaluation
│   └── visualize.py                # Dashboard & plot generation
├── data/
│   ├── unified_queries.parquet     # Query-level execution times
│   └── unified_operators.parquet   # Operator-level metrics (~138K rows)
└── results/
    ├── sigma_model_v3.pkl          # Trained model (79 features, 500 trees)
    ├── hinted_queries_xeon_multi_source/  # Multi-Source Xeon recommendations
    ├── hinted_queries_epyc_v2/           # EPYC recommendations
    └── figures/                          # Evaluation plots
```

---

## 10. Running HALO

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step 1: Parse experiment logs → Parquet
```bash
python3 src/explain_parser.py
# Output: data/unified_operators.parquet, data/unified_queries.parquet
```

### Step 2: Train σ model
```bash
python3 src/sigma_model_v3.py
# Output: results/sigma_model_v3.pkl (79 features, ~138K training samples)
```

### Step 3: Generate recommendations (Multi-Source)
```bash
python3 src/generate_hinted_queries.py
# Output: results/hinted_queries_xeon_multi_source/{NVMe,SATA}/{TPCH,JOB}/*.sql
```

### Step 4: Register a new target server
```python
from hw_registry import register_env
register_env('CloudVM_NVMe', {
    'server': 'CloudVM', 'storage': 'NVMe', 'cpu': 'EPYC-9654',
    'seq_read_mbps': 7000, 'rand_read_iops': 1500000,
    'seq_write_mbps': 5000, 'rand_write_iops': 500000,
    'cpu_cores': 96, 'cpu_threads': 192,
    'cpu_base_ghz': 2.4, 'cpu_boost_ghz': 3.7, 'l3_cache_mb': 384,
    'ram_gb': 512, 'buffer_pool_gb': 100,
})
```

### Step 5: Run full evaluation
```bash
python3 src/evaluate_halo.py     # Cross-environment comparison
python3 src/visualize.py         # Generate all figures
```

---

## 11. Key Differentiation

| Aspect | Traditional Approach | HALO |
|--------|---------------------|------|
| **Granularity** | Query-level prediction | **Operator-level** (each JOIN, SCAN, SORT independently assessed) |
| **HW Awareness** | None or binary flags | **Quantitative**: IOPS, MB/s, GHz, cores, cache, RAM |
| **Knowledge Source** | Single server | **Multi-Source Global Ensemble** across all known servers |
| **Risk Management** | Apply or don't | **3-Tier confidence** with Risk-Reward EV calculation |
| **Interaction Modeling** | Independent features | **Cross-terms**: `scan × IOPS_ratio`, `join × CPU_clock_ratio` |

---

## 12. Citation
```
HALO v3: Hardware-Aware Learned Optimizer with Multi-Source Global Knowledge Ensemble
Big Data Lab Research, 2026
```
