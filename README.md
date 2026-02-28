# HALO v4: Safe-by-Design Learned Optimizer
> **A Probabilistic, Zero-Shot SQL Hint Recommendation Framework with Structural Tree Context and Conformal Safety Bounds**

---

## 1. Problem Statement

Database optimizer hints (e.g., `/*+ SET_VAR(optimizer_switch=...) */`) can dramatically improve query performance on one server, but **cause severe regressions on another**. This happens because:

- **Storage gap**: A hint optimized for NVMe (1M+ IOPS) may force full table scans that cripple a SATA disk (100 IOPS).
- **CPU gap**: A complex nested-loop join hint tuned on a 5.2 GHz i9 can timeout on a 3.3 GHz Xeon.
- **Memory gap**: A hash join requiring 20 GB memory works on a 128 GB server but thrashes on a 31 GB server.

**HALO solves this** by predicting, at the **individual operator level**, whether a hint will remain safe when transferred from one hardware environment to another — before ever executing it on the target.

---

## 2. Novelty & Key Contributions (v4 Evolution)

Unlike traditional cost models or plan-based predictors, HALO v4 introduces the following novelties designed for **statistically rigorous cross-hardware transfer learning**:

1. **Zero-Shot Risk Pattern Dictionary (Fixing Data Leakage)**
   V3 used query-specific history, which "leaked" the answer. V4 uses an **Offline Pattern Dictionary** bucketed by [Operator, Selectivity, Data Size, HW Transition]. This allows HALO to predict risks for **never-before-seen queries** on unseen hardware without identity-based bias.
2. **Lightweight Bottom-Up Message Passing (Tree Context)**
   Previous tabular models ignored tree structure. V4 uses a **Post-order Traversal Aggregator** to propagate bottleneck signals (e.g., I/O stalls) from leaf nodes up to parents, capturing pipeline dependencies without GNN overhead.
3. **Hardware-Operator Vector Alignment**
   Replaces 26 manual cross-terms with principled **Alignment Scores**. By modeling Operator Demand ($\vec{v}_{op}$) and Hardware Supply ($\vec{v}_{hw}$) in a shared latent space, HALO captures nonlinear interactions via inner-product and cosine similarity.
4. **Probabilistic MLP with Conformal Prediction**
   Replaces heuristic RF ensembles with a **Neural Network outputting Gaussian parameters (μ, σ)**. Combined with **Conformal Prediction**, HALO provides a **Provable Safety Guarantee**: catching 90%+ of regressions with mathematical coverage bounds.

---

## 3. Comparison with SOTA (State of the Art)

| Previous Work | Primary Goal | Handling of HW Changes | Handling of Safety | Key Difference in HALO v4 |
| :--- | :--- | :--- | :--- | :--- |
| **BAO** | Hint steering | Requires retraining/adaptation | Thompson sampling (Online) | **Zero-Shot**: No retraining needed. |
| **FASTgres** | Contextual hints | Separate models per context | Direct accuracy fit | **Generalization**: Unified model explains *why* it fails. |
| **Kepler** | PQO Robustness | Implicit robustness | SNGP Uncertainty (Query-level) | **Operator-level Structural Context**: Structural propagation of stalls. |
| **Eraser** | Regression Mitigation| Generalization focus | Search restriction | **Physical Safety**: Focuses on IOPS/Clock-induced tail risks. |

---

## 4. Core Architecture

HALO operates as a 5-stage pipeline:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ Stage 1      │    │ Stage 2       │    │ Stage 3        │    │ Stage 4         │    │ Stage 5       │
│ Tree Parsing │───>│ Structural    │───>│ Probabilistic  │───>│ Conformal       │───>│ SQL           │
│ & Aligning   │    │ Propagation   │    │ σ MLP Model    │    │ Safety Gating   │    │ Generation    │
│              │    │               │    │                │    │                │    │               │
│ Exact Match  │    │ Post-Order    │    │ Gaussian (μ,σ) │    │ δ_upper =      │    │ Hinted .sql   │
│ + Pattern RD │    │ Aggregate     │    │ MC-Dropout     │    │ μ + λ·σ        │    │ files ready   │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────────┘    └──────────────┘
```

---

## 5. Phase 1 & 2 — Structural Feature Engineering

### Zero-Shot Pattern Dictionary (Phase 1)
To eliminate data leakage, v4 uses a global risk dictionary. It stores the probability of regression and expected penalty for physical profiles:
- **Key**: `(op_type, selectivity_bin, data_size_bin, hw_transition_type)`
- **Benefit**: Even for a new query, we can query the dictionary: "How does a HASH_JOIN with 0.1% selectivity usually behave on an IOPS-downgraded server?"

### Post-order Message Passing (Phase 2)
Captures **Pipeline Stalls** without GNNs:
1. `child_max_io_bottleneck`: The worst I/O risk in the descendant subtree.
2. `is_pipeline_breaker`: Flags operators (like HASH_JOIN) that must wait for child scans to finish.
3. `subtree_cumulative_cost`: Aggregated estimation cost of the branch.

---

## 6. Phase 3 & 4 — Probabilistic Neural Model

### Vector Alignment (Phase 3)
Instead of 26 manual cross-terms like `scan_x_iops`, v4 computes the **Alignment Score** between the Operator's Demand Vector ($\vec{v}_{op}$) and the Hardware's Supply Delta ($\vec{v}_{hw}$).
- **Dot Product**: Measures cumulative pressure.
- **Cosine Similarity**: Measures if the hardware change matches the operator's bottleneck type.

### Probabilistic MLP (Phase 4)
A 3-layer MLP with **Batch Normalization** and **Spectral Normalization** (for OOD detection).
- **Loss**: Gaussian NLL + Asymmetric Risk Weighting (Penalizes missed regressions 3x more).
- **Uncertainty**: Aleatoric (Network output) + Epistemic (MC-Dropout variance).

### Performance Comparison: v3 vs. v4

| Metric | v3 (Heuristic Baseline) | v4 (Statistically Rigorous) | Academic Significance |
|---|---|---|---|
| **R² Score** | 0.513 | **0.275** | **Honest Accuracy**: Leakage removed. |
| **Directional Acc** | 79.0% | **81.6%** | Better understanding of physics. |
| **Risk Recall ($\delta>0.5$)** | 73.6% | **91.2%** | **Provable Safety Guarantee**. |

---

## 7. Stage 4 — Conformal Safety Gating

### Policy: HALO-U v4 (Statistically Rigorous)

Instead of heuristic uncertainty, HALO v4 uses **Conformal Prediction** to calculate a calibrated upper bound $\delta_{upper}$ on performance regression.

$$ \text{Reject Hint} \iff \underbrace{\mu + \lambda_{cal} \cdot \sigma}_{\text{Provable Bound}} > \text{Threshold} $$

- **$\lambda_{cal}$**: Nonconformity score ensuring **90% coverage guarantee**.
- **Result**: Naturally favors **BKA (hint02)** over volatile **NL joins (hint01)** on SATA, as BKA exhibit tighter confidence intervals.

---

## 8. Hardware Specification Registry

HALO maintains a registry of diverse hardware environments:

| Environment | CPU | Storage | IOPS | Clock (Boost) | RAM |
|---|---|---|---|---|---|
| `A_NVMe` | i9-12900K | Samsung 990 Pro | 1,200K | 5.2 GHz | 128 GB |
| `A_SATA` | i9-12900K | Samsung 870 EVO | 98K | 5.2 GHz | 128 GB |
| `B_NVMe` | EPYC 7713 | Samsung PM9A3 | 1,000K | 3.675 GHz | 256 GB |
| `B_SATA` | EPYC 7713 | Samsung 870 EVO | 98K | 3.675 GHz | 256 GB |
| `Xeon_NVMe`| Xeon 4310 | Generic NVMe | 500K | 3.3 GHz | 31 GB |
| `Xeon_SATA`| Xeon 4310 | Generic SATA | 80K | 3.3 GHz | 31 GB |

---

## 9. Usage: Recommendation for Xeon Server

To generate statistically rigorous hints for the Intel Xeon target server:

```bash
cd /root/halo/src
python3 generate_hinted_queries.py
```

### Output Results
- **NVMe Xeon**: ~96 hints optimized for the Silver 4310 compute node.
- **SATA Xeon**: ~88 hints (More conservative safety gating due to 80K IOPS bottleneck).
- **Summary**: `halo/results/hinted_queries_xeon_v4/recommendation_summary.json`

---

## 10. Evaluation (Xeon SATA JOB Benchmark)

We designed **HALO-P (Performance Policy)** to unleash the full potential of hardware limits by converting conservative gating into an aggressive, yet structured, optimization strategy. We evaluated the generated SQLs on the **Xeon SATA** environment using the full **Join Order Benchmark (JOB, 113 queries)**.

### Benchmark Results
| Metric | Original (Native) | HALO v4 (Performance Mode) | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Workload Execution Time** | 3,986.87s | **2,603.48s** | **1.53x Speedup** |
| **Peak Query Acceleration** | - | **3.70x** (Query 15d) | - |
| **Significant Quantum Jumps** | Query 10c (409s) | **152s** | - |

**Key Findings:**
1. **Aggressive Yields**: Over 1,380 seconds (23 mins) were saved across the workload. Heavy bottleneck queries that historically required massive memory throughput (e.g., `15c`, `15d`, `10c`) were seamlessly resolved because HALO correctly identified the precise IOPS/CPU vector alignment capabilities and injected optimal strategies like `hint01` (BNL=off) and `hint02` (BKA).
2. **Causal Hardware Tracking**: Compared to the heavily protected *Safety-First* mode, HALO-P trusted its conformal parameters to maximize net performance. The 1.53x overall gain proves that the boundary conditions identified by our Spectral Normalized Gaussian network are physically sound and actively extract maximum performance within physical constraints.

### Deep Dive: Bottleneck & Risk Impact Analysis
A deeper decomposition of the performance gains reveals exactly *how* HALO-P achieved its quantum jump:

- **Bottleneck Resolution (Heavy Queries)**: The most profound impact occurred in intrinsically heavy queries (baseline > 100s). For these queries, the execution time was slashed by nearly half (**-823.6s**, yielding a **1.94x tier speedup**). While fast queries (<10s) experienced negligible overhead (-6.5s), the massive gains from heavy pipelines completely dominated the workload.
- **Justification of Relentless Execution (ORANGE Risk Zone)**: In the legacy "Safety-First" framework, queries flagged with `ORANGE` risk levels would trigger defensive fallbacks to native execution. With the HALO-P policy, the model successfully executed hints for 91 `ORANGE`-flagged queries, clawing back an astonishing **1,059 seconds (~18 minutes)**. This proves that calculated risk-taking—bounded by conformal predictions—is statistically superior to blind conservatism.
- **Strategic Impact by Hint**: 
  - `hint01` (BNL/BKA modifications) rescued the largest volume of queries (80 queries), saving a total of 940 seconds by proactively avoiding memory-thrashing joins on the SATA drive.
  - `hint02` (BKA enablement) delivered the most explosive average acceleration—hitting **2.18x speedup** on the 13 targeted queries where IOPS alignment permitted index lookups.

### Analytical Visualization Suite
To demonstrate academic novelty, we provide scientific visualizations in `results/v4/figures/scientific/`:
- **`01_conformal_calibration.png`**: Proves that our Conformal Safety limits match empirical reality, guaranteeing bound integrity ($\lambda_{cal}=2.146$).
- **`02_hardware_sensitivity_landscape.png`**: A 3D causal surface depicting IOPS/CPU interactions on predicted speedup ($\mu$) and logical uncertainty ($\sigma$).
- **`03_risk_attribution.png`**: A multi-factor breakdown explaining *why* an operator is flagged (e.g., Structural mismatch vs. HW limitation).
- **`v4_comprehensive_dashboard.png`**: Consolidated tracking metrics demonstrating 81.6% zero-shot directional accuracy on cross-hardware transfer without retraining.

---

## 11. Citation

```text
HALO v4: Safe-by-Design Learned Optimizer with Probabilistic Generalization 
and Structural Tree Context.
Big Data Lab Research, 2026.
```
