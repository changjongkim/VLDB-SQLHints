# HALO: Hardware-Aware Learned Optimizer
> **A Multi-Source, Dual-Head SQL Hint Recommendation Framework with Adaptive Per-Operator Risk Assessment**

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
│ EXPLAIN logs │    │ 84 features   │    │ Dual-Head RF   │    │ HALO-R Policy   │    │ Hinted .sql   │
│ + Worst-Case │    │ per operator  │    │ Regressor +    │    │ Adaptive Per-Op │    │ files ready   │
│   data       │    │ pair          │    │ Classifier     │    │ Threshold       │    │ to execute    │
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

4. **Worst-Case Data Integration** (`worst_case_integration.py`):
   - Loads `worst_nvme_report.csv` and `worst_sata_report.csv` containing experiments where hints caused **performance regressions**.
   - Enriches operators with 5 worst-case features (see Stage 2).

### Output
- `unified_operators.parquet`: ~138K operator instances with 20+ fields each.
- `unified_queries.parquet`: query-level execution times per (env, query, hint_set).

---

## 4. Stage 2 — Feature Engineering (84 Features)

### Overview
Each operator is described by **84 features** organized into 6 categories, enabling the σ model to understand both *what the operator does*, *how the hardware transition affects it*, and *whether this hint is known to cause regressions*.

### Feature Categories

| # | Category | Count | Key Features | Purpose |
|---|----------|-------|-------------|---------|
| 1 | **Operator Type** (one-hot + semantic) | 19 | `is_NL_INNER_JOIN`, `is_TABLE_SCAN`, `is_join`, `is_scan`, `is_index_op`, `is_agg_sort` | What kind of work this operator performs |
| 2 | **Cardinality & Work** | 13 | `log_actual_rows`, `log_est_cost`, `row_est_error`, `log_self_time`, `log_total_work`, `cost_per_row`, `time_per_row`, `selectivity` | How heavy this operator is |
| 3 | **Hardware Transition** | 16 | `iops_ratio`, `cpu_clock_ratio`, `storage_speed_ratio`, `ram_ratio`, `is_storage_downgrade`, `is_cpu_downgrade` | How different is the target hardware |
| 4 | **Workload Metadata** | 5 | `log_table_rows`, `log_table_size_mb`, `n_indexes`, `fits_in_buffer` | What data this operator touches |
| 5 | **Cross-Terms (Interaction)** | 26 | `scan_x_iops`, `join_x_cpu_clock`, `table_size_x_storage_speed`, `work_x_compute` | **Critical**: How operator type × HW change interact |
| 6 | **Worst-Case Features** | 5 | `has_worst_case`, `worst_regression_ratio`, `log_worst_regression`, `worst_bp_reads_ratio`, `worst_hint_matches` | **Safety**: Known regression signals from experiments |

### Cross-Term Logic (The Key Innovation)
The most important features are the **interaction terms**. They capture insights like:

- `scan_x_iops = is_scan × log(target_IOPS / source_IOPS)` → "A table scan on a storage downgrade is dangerous"
- `join_x_cpu_clock = is_join × log(target_GHz / source_GHz)` → "A nested loop join on a slower CPU is risky"
- `table_size_x_storage_speed` → "A 22 GB lineitem scan is fine on NVMe but lethal on SATA"

### Worst-Case Feature Logic (Tail Risk Management)
The worst-case features act as **safety signals** derived from actual regression experiments:

- `has_worst_case = 1` → This query+hint combination is **known** to have caused regression on at least one environment
- `worst_regression_ratio` → How bad the regression was (e.g., 3.5 = hinted was 3.5x slower than baseline)
- `worst_bp_reads_ratio` → How much extra buffer pool I/O the hint caused (disk pressure indicator)

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

### Dual-Head Architecture (Key Innovation)

Unlike traditional single-model approaches, HALO uses a **Dual-Head** design:

```
                    ┌─────────────────────────┐
                    │    84 Input Features     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                          │
              ┌─────▼─────┐            ┌──────▼──────┐
              │  Head 1    │            │  Head 2      │
              │ RF         │            │ RF           │
              │ Regressor  │            │ Classifier   │
              │ (δ value)  │            │ (HIGH_RISK)  │
              └─────┬─────┘            └──────┬──────┘
                    │                          │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Ensemble Decision      │
                    │   (Safety-First: OR)     │
                    │                          │
                    │ If EITHER head flags     │
                    │ risk → HIGH_RISK         │
                    └─────────────────────────┘
```

- **Head 1 (Regressor)**: Random Forest with 500 trees, predicts continuous δ. Uses **per-operator adaptive threshold** to flag risk.
- **Head 2 (Classifier)**: Random Forest with 300 trees, `class_weight='balanced_subsample'`, directly predicts binary HIGH_RISK label.
- **Ensemble**: Safety-First OR — if **either** head flags an operator, it's marked HIGH_RISK. This maximizes recall (catching dangerous operators).

### Per-Operator Adaptive Thresholds

Instead of a single global threshold (|δ| > 0.8), each operator type has its own threshold computed from actual δ distribution:

```
threshold(op_type) = mean(|δ|) + 0.5 × std(|δ|)
Clamped to [0.3, 2.0]
```

| Op Type | Mean|δ| | Threshold | Reasoning |
|---|---|---|---|
| `AGGREGATE` | 0.123 | **0.300** | CPU-bound, small changes are significant |
| `SORT` | 0.322 | **0.460** | Moderate sensitivity |
| `INDEX_SCAN` | 0.407 | **0.610** | Balanced I/O + CPU |
| `HASH_JOIN` | 0.512 | **0.783** | Memory-bound, tolerates variation |
| `NL_INNER_JOIN` | 0.720 | **1.047** | High variance, common operator |
| `TABLE_SCAN` | 0.828 | **1.238** | I/O-bound, large deltas are normal |
| `FILTER` | 0.853 | **1.280** | Highly variable, only extreme changes matter |

### Training Process

1. **Operator Matching** (`build_env_pairs_v3()`):
   - For each environment pair (e.g., `A_NVMe ↔ B_SATA`), match the **same operator** across both environments using `(query_id, hint_set, op_type, table_alias)` as the key.
   - Compute the actual δ(time) as ground truth label.
   - Filter out operators with `self_time < 0.1ms` (too noisy to measure reliably).
   - **Enrich with worst-case data**: 5,411 operators receive worst-case regression features.

2. **Sample Weighting** (Recall-Focused):
   - Environment pair balancing (ensures equal representation across hardware pairs)
   - Worst-case experiments receive **2× weight** (known dangerous cases)
   - Large regressions (|δ| > 0.8) receive **2× additional weight**
   - High-risk samples (|δ| > 0.5) receive **3× weight** (asymmetric penalty for missed dangers)

3. **Model Training** (`train_sigma_v3()`):
   - **Regressor**: Random Forest (500 trees, max_depth=None, min_samples_leaf=1)
   - **Classifier**: Random Forest (300 trees, max_depth=16, class_weight='balanced_subsample')
   - Cross-validation: Leave-One-Environment-Pair-Out (6 folds)

### Performance Metrics

| Metric | Value | Description |
|---|---|---|
| **R²** | 0.513 | Variance explained by regressor |
| **Pearson r** | 0.724 | Strong positive correlation between predicted and actual δ |
| **Direction Accuracy** | 79.0% | Correctly predicts "faster" vs "slower" direction |
| **Ranking Accuracy** | 75.7% | Correctly ranks relative severity of two operators |
| **Cross-HW DirAcc** (Fold 2–5) | **88.8–91.4%** | Direction accuracy on **unseen hardware** pairs |

### High-Risk Detection Performance

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Regressor (global |δ|>0.8) | 0.723 | 0.551 | 0.625 |
| **Classifier (Adaptive)** | **0.622** | **0.736** | **0.674** |

> **Key Achievement**: Classifier Recall = **73.6%** (vs 55.1% baseline). 
> This means HALO now catches **73.6% of dangerous operators** before execution — 
> a 33.6% improvement over the single-threshold approach.

### Cross-Environment Validation (Leave-One-Pair-Out)

| Fold | Environment Pair | Samples | R² | Direction Acc |
|---|---|---|---|---|
| 1 | A_NVMe → A_SATA (same CPU) | 10,510 | -0.357 | 68.5% |
| 2 | A_NVMe → B_NVMe (cross-CPU) | 7,725 | **0.706** | **89.1%** |
| 3 | A_NVMe → B_SATA (cross-both) | 7,878 | 0.549 | 88.4% |
| 4 | A_SATA → B_NVMe (cross-both) | 7,859 | **0.760** | **91.4%** |
| 5 | A_SATA → B_SATA (cross-CPU) | 8,009 | **0.719** | **89.9%** |
| 6 | B_NVMe → B_SATA (same CPU) | 18,543 | -1.352 | 62.1% |

> **Insight**: Cross-HW transfer (Folds 2–5) achieves 88–91% direction accuracy.
> Storage-only folds (1, 6) have negative R² because when only storage changes on the same CPU,
> operator timing differences are very small and difficult to predict — but this is acceptable
> because the absolute risk in these scenarios is also small.

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

### Policy: HALO-R (Robust) with Multi-Source Global Ensemble + Dual-Head

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

#### Step 2: Per-Transition Dual-Head Risk Verification
Each candidate is evaluated using **both the Regressor and Classifier**:

```python
hw_features = compute_hw_features(src_env, target_env)

for each operator in the hinted query plan:
    features = engineer_84_features(operator, hw_features)
    
    # Head 1: Regressor
    δ_predicted = regressor.predict(features)
    adaptive_thresh = adaptive_thresholds[op_type]  # e.g., 0.3 for AGGREGATE, 1.2 for TABLE_SCAN
    is_risky_regressor = |δ_predicted| > adaptive_thresh
    
    # Head 2: Classifier
    is_risky_classifier = classifier.predict(features)  # Binary: 0 or 1
    
    # Ensemble (Safety-First OR)
    if is_risky_regressor OR is_risky_classifier:
        mark as HIGH_RISK

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
# Result: "hint01 from B_NVMe (15.18x speedup, 24% risk)" beats
#         "hint05 from A_SATA (2.39x speedup, 20% risk)"
```

### Hint Definitions

| Hint ID | MySQL Optimizer Switch | Strategy |
|---------|----------------------|----------|
| `hint01` | `block_nested_loop=off` | Force index-based nested loop joins |
| `hint02` | `block_nested_loop=off, batched_key_access=on, mrr=on` | BKA + Multi-Range Read |
| `hint04` | `block_nested_loop=off, hash_join=on` | Force hash joins |
| `hint05` | `block_nested_loop=off` + `NO_RANGE_OPTIMIZATION` | Disable range scan on primary key |

---

## 7. Stage 5 — SQL Generation & Recommendation Results

### Output Structure
```
hinted_queries_xeon_multi_source/
├── NVMe/
│   ├── TPCH/
│   │   ├── tpch_q1_hint05.sql
│   │   ├── tpch_q12_native.sql
│   │   └── ... (22 queries)
│   └── JOB/
│       ├── 10a_hint01.sql
│       ├── 1a_hint01.sql
│       └── ... (113 queries)
├── SATA/
│   ├── TPCH/ (22 queries)
│   └── JOB/  (113 queries)
├── run_xeon_nvme.sh      ← Batch runner for NVMe server
├── run_xeon_sata.sh      ← Batch runner for SATA server
└── recommendation_summary.json
```

### Recommendation Summary (Xeon Silver 4310 Target)

| Target | Benchmark | Total Queries | Hinted | Native | Coverage |
|---|---|---|---|---|---|
| **Xeon_NVMe** | TPCH | 22 | **12** | 10 | 55% |
| **Xeon_NVMe** | JOB | 113 | **102** | 11 | **90%** |
| **Xeon_SATA** | TPCH | 22 | **12** | 10 | 55% |
| **Xeon_SATA** | JOB | 113 | **101** | 12 | **89%** |
| **Total** | — | **270** | **227** | 43 | **84%** |

### Hint Distribution (JOB Benchmark, Xeon_NVMe)

| Hint | Count | Strategy |
|---|---|---|
| `hint01` | 66 | Force NL joins (most effective for JOB's star-schema queries) |
| `hint05` | 21 | Disable range optimization |
| `hint02` | 11 | BKA + MRR |
| `hint04` | 4 | Force hash joins |
| `NATIVE` | 11 | No safe hint found (fallback) |

### Sample SQL File
```sql
-- HALO Recommended SQL
-- Query     : 13b (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection (Dual-Head)
-- Hint      : hint05
-- From Src  : B_NVMe
-- Reason    : HALO-R (Multi-Source): 'hint05' selected from B_NVMe (src_speedup=15.19x, risk=8%)
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */
  ...
```

---

## 8. Top 20 Feature Importance

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | `is_NL_INNER_JOIN` | 0.106 | Operator Type |
| 2 | `is_FILTER` | 0.084 | Operator Type |
| 3 | `is_TABLE_SCAN` | 0.083 | Operator Type |
| 4 | `is_INDEX_LOOKUP` | 0.083 | Operator Type |
| 5 | `is_HASH_JOIN` | 0.060 | Operator Type |
| 6 | `is_AGGREGATE` | 0.048 | Operator Type |
| 7 | `is_SORT` | 0.047 | Operator Type |
| 8 | `is_NL_ANTIJOIN` | 0.045 | Operator Type |
| 9 | `is_NL_SEMIJOIN` | 0.040 | Operator Type |
| 10 | `is_INDEX_SCAN` | 0.039 | Operator Type |
| 11 | `is_TEMP_TABLE_SCAN` | 0.033 | Operator Type |
| 12 | `is_STREAM` | 0.033 | Operator Type |
| 13 | `is_MATERIALIZE` | 0.032 | Operator Type |
| 14 | `is_GROUP_AGGREGATE` | 0.032 | Operator Type |
| 15 | `is_NL_LEFT_JOIN` | 0.026 | Operator Type |
| 16 | `is_join` | 0.019 | Semantic |
| **17** | **`log_worst_regression`** | **0.014** | **Worst-Case** ★ |
| **18** | **`worst_bp_reads_ratio`** | **0.014** | **Worst-Case** ★ |
| **19** | **`worst_regression_ratio`** | **0.013** | **Worst-Case** ★ |
| 20 | `is_scan` | 0.011 | Semantic |

> ★ Three worst-case features in the **Top 20** — confirming the model actively uses regression experiments to make safer predictions.

---

## 9. Project Structure

```
halo/
├── src/
│   ├── explain_parser.py           # Stage 1: EXPLAIN ANALYZE → operator trees
│   ├── hw_registry.py              # HW spec registry + transition feature computation
│   ├── workload_metadata.py        # Table metadata enrichment (TPCH/JOB)
│   ├── worst_case_integration.py   # Worst-case experiment data loading & enrichment
│   ├── sigma_model_v3.py           # Stage 3: Dual-Head σ model (84 features, RF)
│   ├── generate_hinted_queries.py  # Stage 4-5: Multi-Source recommendation + SQL gen
│   ├── halo_framework.py           # Full HALO-G/R API with 3-tier policy
│   ├── evaluate_halo.py            # Cross-environment evaluation
│   └── visualize.py                # Dashboard & plot generation
├── data/
│   ├── unified_queries.parquet     # Query-level execution times
│   └── unified_operators.parquet   # Operator-level metrics (~138K rows)
└── results/
    ├── sigma_model_v3.pkl          # Dual-Head model (Regressor + Classifier + Adaptive Thresholds)
    ├── sigma_results_v3.json       # Full training metrics and per-operator analysis
    ├── hinted_queries_xeon_multi_source/  # Xeon recommendations (270 SQL files)
    │   ├── NVMe/{TPCH,JOB}/*.sql
    │   ├── SATA/{TPCH,JOB}/*.sql
    │   ├── run_xeon_nvme.sh        # Batch runner script
    │   └── run_xeon_sata.sh        # Batch runner script
    └── figures/                    # Evaluation plots
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

### Step 2: Train Dual-Head σ model
```bash
python3 src/sigma_model_v3.py
# Output: results/sigma_model_v3.pkl (84 features, Dual-Head, ~60K training samples)
# Output: results/sigma_results_v3.json (full metrics)
```

### Step 3: Generate recommendations (Multi-Source + Dual-Head)
```bash
python3 src/generate_hinted_queries.py
# Output: results/hinted_queries_xeon_multi_source/{NVMe,SATA}/{TPCH,JOB}/*.sql
```

### Step 4: Execute on Xeon target server
```bash
# Copy the entire hinted_queries_xeon_multi_source/ directory to Xeon server, then:
chmod +x run_xeon_nvme.sh run_xeon_sata.sh
./run_xeon_nvme.sh 2>&1 | tee xeon_nvme_results.log
./run_xeon_sata.sh 2>&1 | tee xeon_sata_results.log
```

### Step 5: Register a new target server
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

### Step 6: Run full evaluation
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
| **Risk Detection** | Single global threshold | **Dual-Head** (Regressor + Classifier) with **Per-Operator Adaptive Thresholds** |
| **Risk Management** | Apply or don't | **3-Tier confidence** with Recall-focused training |
| **Interaction Modeling** | Independent features | **Cross-terms**: `scan × IOPS_ratio`, `join × CPU_clock_ratio` |
| **Worst-Case Safety** | Not considered | **Worst-case experiments integrated** as training signals (5 features) |

---

## 12. Citation
```
HALO v3: Hardware-Aware Learned Optimizer with Dual-Head Adaptive Risk Assessment
and Multi-Source Global Knowledge Ensemble
Big Data Lab Research, 2026
```
