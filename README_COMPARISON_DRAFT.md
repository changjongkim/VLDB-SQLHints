## 3. Comparison with SOTA (State-of-the-Art)

While recent literature has explored learned query optimization and hint recommendation, HALO distinguishes itself by treating **Cross-Hardware Transferability** and **Safety against Performance Regressions** as first-class citizens.

| Previous Work (Type) | Primary Goal / Output | Handling of Hardware Changes | Handling of Regressions / Safety | Key Difference in HALO |
| :--- | :--- | :--- | :--- | :--- |
| **BAO** (Learned Steering) | Recommend hint sets per query | Requires retraining/online adaptation for new HW. Does not decompose cross-HW traits. | Relies on Thompson sampling to adapt over time. | **Transfer-First**: HALO transfers hints learned on one HW to another *without* retraining, by modeling operator-level HW sensitivity. |
| **FASTgres** (Workload/HW Modeling) | Predict/Classify hints per context | Trains isolated models per context (HW/Version/Workload). | Aims to fit the current context accurately. | **Zero-Shot Transfer**: HALO doesn’t build separate models per HW. It models *why* (via HW cross-terms) a plan breaks, enabling safe zero-shot transfer. |
| **Kepler** (PQO Robustness) | Robust plan selection for Parametric Queries | Handled implicitly via plan robustness within the same environment. | Uses Spectral-normalized Gaussian Processes (SNGP) for uncertainty; falls back to default if uncertain. | **Operator-Level Uncertainty**: Kepler applies uncertainty to query-level plan selection. HALO uses Random Forest variance at the **operator-level** specifically for cross-HW penalties (HALO-U). |
| **Eraser** (Regression Mitigation) | Plugin to eliminate regressions in Learned Optimizers | Focuses on model generalization failures, not HW transitions. | Restricts search space or overrides learned models when regressions are detected. | **Hardware-Induced Regressions**: HALO focuses specifically on regressions caused by physical HW differences (e.g., IOPS) turning a good hint into poison. |
| **Roq / COOOL** (Risk/Rank-aware QO) | Rank robust plans using expected cost + risk | Not primarily focused on cross-hardware transfer. | Incorporates risk predictions into the cost model directly. | **Intervention as a Choice**: HALO predicts whether the specific operators mutated by a hint become dangerous, deciding whether to intervene or fallback. |

### Summary of Core Contributions (C1–C5)

**C1. Systematization of the Hint Portability Problem**
HALO redefines hint recommendation not as "optimization within a single environment," but as the **safe transfer of knowledge to differing hardware**. It systematically quantifies why transferring a hint often causes severe regressions, advancing beyond simple observation of HW dependency (e.g., FASTgres).

**C2. Operator-Level Hardware Sensitivity Learning (σ/δ)**
Instead of treating query execution as a black box (like BAO), HALO decomposes query plans and predicts performance changes and risks for the specific operators altered by a hint. By multiplying operator semantics with hardware deltas, HALO explains *which operator* becomes dangerous and *why*.

**C3. Dual-Head Safety-First Architecture for Tail Risk Optimization**
Predicting severe performance regressions (tail risks) suffers from extreme class imbalance. HALO solves this by combining a Regressor (predicting actual time deltas) with a dedicated Classifier (heavily weighted to catch dangerous regressions). If *either* head flags a risk, the intervention is blocked.

**C4. Uncertainty-Aware Conservative Gating (HALO-U)**
While works like Kepler utilize uncertainty for fallback, HALO applies it to **cross-hardware transfer at the operator level**. By using the natural variance across its Random Forest estimators as a penalty measure, the HALO-U policy heavily discounts strategies that are highly uncertain on new hardware, naturally converging on stable strategies without human-crafted heuristics.

**C5. Structural Integration of Worst-Case Knowledge**
Similar in spirit to Eraser's regression mitigation, HALO inherently loops "past worst-case regression experiences" directly into its feature space and sample weighting. This forces the model to heavily penalize historically disastrous operator-hardware combinations, directly optimizing tail-risk safety.
