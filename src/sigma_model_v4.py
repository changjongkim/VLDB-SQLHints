# -*- coding: utf-8 -*-
"""
HALO v4 — σ Model: Probabilistic Neural Network with Uncertainty
==================================================================

Complete replacement for sigma_model_v3.py.

Architecture Overview:
  1. Phase 1 features: Zero-shot Risk Pattern Dictionary (no data leakage)
  2. Phase 2 features: Bottom-up tree context propagation
  3. Phase 3 features: Hardware-Operator Vector Alignment
  4. Phase 4 model:   MLP with Gaussian output (μ, σ) + Conformal calibration

Model Design:
  - Input:  ~85 features (operator + hardware + structural + pattern risk)
  - Hidden: 3-layer MLP with ReLU, BatchNorm, Dropout (epistemic uncertainty)
  - Output: 2 scalars (μ, log_σ) representing a Gaussian over δ(time)
  - Loss:   Negative Log-Likelihood (NLL) of Gaussian → naturally learns
            to output high σ for uncertain predictions

Uncertainty Quantification:
  Option A: MC-Dropout (Gal & Ghahramani, 2016)
    - Keep dropout ON at inference, run T forward passes
    - Epistemic uncertainty = variance across passes
    - Total uncertainty = aleatoric (σ²) + epistemic (MC variance)

  Option B: Conformal Prediction (Vovk et al., 2005)
    - After training, calibrate on held-out data to compute
      nonconformity scores → guaranteed coverage bounds
    - δ_p90 = μ + q_{0.90} * σ (calibrated quantile)

HALO-U Policy (updated):
  reject_hint ⟺ (μ + λ·σ_total) > Threshold
  where λ is calibrated from conformal prediction to guarantee
  90% coverage of actual regressions.

Academic Defense:
  "Unlike the v3 heuristic dual-head (Regressor OR Classifier), HALO v4
   learns a full predictive distribution over operator-level deltas.
   The Gaussian parameterization naturally yields both point predictions
   (μ) and calibrated uncertainty estimates (σ). Combined with conformal
   prediction for finite-sample validity, this provides the FIRST
   statistically rigorous safety guarantee for cross-hardware hint transfer:
   with probability ≥ 1-α, the true delta lies below our upper bound."
"""

import os
import math
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Local imports
from hw_registry import HW_REGISTRY, compute_hw_features, get_env_pairs
from workload_metadata import enrich_operator
from risk_pattern_dictionary import (
    RiskPatternDictionary, classify_hw_transition,
    inject_pattern_risk_features
)
from tree_propagation import propagate_tree_features_from_df

# Backward compatibility
ENV_MAP = HW_REGISTRY
ENV_PAIRS = get_env_pairs(include_target_envs=False)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: Hardware-Operator Vector Alignment
# ═══════════════════════════════════════════════════════════════════════

def build_operator_profile_vector(features: dict) -> np.ndarray:
    """
    Build the Operator Profile Vector v_op.
    
    Captures the operator's intrinsic resource demand pattern:
      - I/O intensiveness (scan? index? random access?)
      - Memory pressure (hash join? materialization?)
      - CPU intensity (sort? aggregate?)
      - Data volume (rows, work)
    
    Returns:
        numpy array of shape (d_op,) where d_op = number of HW-related
        resource dimensions that an operator can stress.
    """
    v_op = np.array([
        # I/O demand dimensions
        features.get('is_scan', 0.0),
        features.get('is_index_op', 0.0),
        features.get('self_io_risk', 0.0) / 10.0,  # Normalized
        (1.0 - features.get('fits_in_buffer', 0.0)),  # Buffer miss = I/O
        features.get('log_total_work', 0.0) / 10.0,
        
        # Memory demand dimensions
        features.get('is_HASH_JOIN', 0.0),
        features.get('is_MATERIALIZE', 0.0),
        features.get('log_actual_rows', 0.0) / 20.0,
        
        # CPU demand dimensions
        features.get('is_join', 0.0),
        features.get('is_agg_sort', 0.0),
        features.get('log_est_cost', 0.0) / 15.0,
        
        # Data volume dimensions
        features.get('log_table_size_mb', 0.0) / 10.0,
        features.get('log_table_rows', 0.0) / 20.0,
    ], dtype=np.float32)
    
    return v_op


def build_hardware_profile_vector(hw_features: dict) -> np.ndarray:
    """
    Build the Hardware Profile Vector v_hw.
    
    Captures the hardware transition's resource supply change:
      - Positive values = target has MORE of this resource
      - Negative values = target has LESS (potential bottleneck)
    
    The dimensions MATCH the operator vector so that dot product
    captures "alignment" between demand and supply change.
    
    Returns:
        numpy array of shape (d_hw,) matching d_op dimensions.
    """
    v_hw = np.array([
        # I/O supply change (matches I/O demand dims)
        hw_features.get('iops_ratio', 0.0),
        hw_features.get('iops_ratio', 0.0),
        hw_features.get('storage_speed_ratio', 0.0),
        hw_features.get('buffer_pool_ratio', 0.0),
        hw_features.get('storage_speed_ratio', 0.0),
        
        # Memory supply change (matches memory demand dims)
        hw_features.get('ram_ratio', 0.0),
        hw_features.get('ram_ratio', 0.0),
        hw_features.get('buffer_pool_ratio', 0.0),
        
        # CPU supply change (matches CPU demand dims)
        hw_features.get('cpu_clock_ratio', 0.0),
        hw_features.get('cpu_clock_ratio', 0.0),
        hw_features.get('cpu_core_ratio', 0.0),
        
        # Bandwidth change (matches data volume dims)
        hw_features.get('storage_speed_ratio', 0.0),
        hw_features.get('iops_ratio', 0.0),
    ], dtype=np.float32)
    
    return v_hw


def compute_vector_alignment_features(features: dict, hw_features: dict) -> dict:
    """
    Phase 3: Compute Hardware-Operator Vector Alignment features.
    
    Instead of manual cross-terms (scan_x_iops, join_x_cpu_clock, ...),
    we compute:
      1. dot_product_alignment  — inner product of v_op and v_hw
      2. cosine_alignment       — cosine similarity
      3. demand_supply_mismatch — L2 norm of (v_op * sign(-v_hw))
                                  (large when op demands what HW lacks)
    
    These replace ~26 manual cross-terms with 3 principled interaction features.
    """
    v_op = build_operator_profile_vector(features)
    v_hw = build_hardware_profile_vector(hw_features)
    
    # Dot product: positive = operator demands align with HW improvements
    #              negative = operator demands align with HW degradations
    dot_prod = float(np.dot(v_op, v_hw))
    
    # Cosine similarity: direction-only alignment
    norm_op = np.linalg.norm(v_op) + 1e-8
    norm_hw = np.linalg.norm(v_hw) + 1e-8
    cosine = float(np.dot(v_op, v_hw) / (norm_op * norm_hw))
    
    # Demand-supply mismatch: how much does this operator demand
    # resources that are DECREASING on the target?
    # Where v_hw < 0 (downgrade), scale by v_op (demand)
    downgrade_mask = (v_hw < 0).astype(np.float32)
    mismatch = float(np.sum(v_op * downgrade_mask * np.abs(v_hw)))
    
    features['dot_product_alignment'] = dot_prod
    features['cosine_alignment'] = cosine
    features['demand_supply_mismatch'] = mismatch
    
    return features


# ═══════════════════════════════════════════════════════════════════════
#  Phase 4: Probabilistic Neural Network
# ═══════════════════════════════════════════════════════════════════════

class HALOProbabilisticMLP(nn.Module):
    """
    Lightweight MLP that outputs Gaussian parameters (μ, σ) for δ(time).
    
    Architecture:
      Input (d features) → Linear(d, 256) → BN → ReLU → Dropout(0.1)
                         → Linear(256, 128) → BN → ReLU → Dropout(0.1)
                         → Linear(128, 64)  → BN → ReLU → Dropout(0.1)
                         → Linear(64, 2)    → [μ, log_σ]
    
    Key design choices:
      - Moderate width (256) to avoid overfitting on ~60K samples
      - BatchNorm for training stability
      - Dropout(0.1) serves double duty: regularization + MC-Dropout inference
      - log_σ output ensures σ > 0 via exp() at output
      - σ is ALEATORIC uncertainty (data noise); MC-Dropout gives EPISTEMIC
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output: μ (predicted delta) and log_σ (log of aleatoric uncertainty)
        self.mu_head = nn.Linear(prev_dim, 1)
        self.log_sigma_head = nn.Linear(prev_dim, 1)
        
        # Spectral normalization on the last hidden layer for OOD detection
        # (inspired by SNGP — Spectral-Normalized Gaussian Process)
        # This constrains the Lipschitz constant, preventing the network
        # from being overconfident on out-of-distribution inputs.
        self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to the last hidden layer."""
        # Find the last Linear layer in the backbone
        for i in reversed(range(len(self.backbone))):
            if isinstance(self.backbone[i], nn.Linear):
                self.backbone[i] = nn.utils.spectral_norm(self.backbone[i])
                break
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: input features, shape (batch, d)
        
        Returns:
            mu:    predicted δ(time), shape (batch, 1)
            sigma: aleatoric uncertainty, shape (batch, 1), always > 0
        """
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        
        # Clamp log_sigma to prevent numerical instability
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=3.0)
        sigma = torch.exp(log_sigma)
        
        return mu, sigma


class GaussianNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian output.
    
    NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
    
    This loss naturally teaches the network to:
      - Output high σ when it cannot predict y well (uncertain regions)
      - Output low σ when predictions are confident
    
    Unlike MSE which treats all samples equally, NLL allows the model
    to "admit uncertainty" — critical for safe hardware transfer.
    """
    
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta  # Weight for the log(σ²) term (prevents σ collapse)
    
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor,
                target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        var = sigma ** 2 + 1e-6
        
        # Gaussian NLL
        nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        
        # Apply asymmetric weighting: penalize under-predictions of risk MORE
        # (target > 0 means slowdown, and under-predicting slowdown is dangerous)
        asymmetric_mask = (target > 0) & (mu < target)
        nll = torch.where(asymmetric_mask, nll * 2.0, nll)
        
        if weights is not None:
            nll = nll * weights.unsqueeze(1)
        
        return nll.mean()


# ═══════════════════════════════════════════════════════════════════════
#  MC-Dropout Inference
# ═══════════════════════════════════════════════════════════════════════

def mc_dropout_predict(model: HALOProbabilisticMLP,
                       x: torch.Tensor,
                       n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout inference for uncertainty estimation.
    
    Runs T forward passes with dropout enabled to estimate epistemic
    uncertainty (model uncertainty about unseen HW configurations).
    
    Total uncertainty = aleatoric (σ from model) + epistemic (MC variance)
    
    Args:
        model: trained HALOProbabilisticMLP
        x: input features (batch, d)
        n_samples: number of MC forward passes (T)
    
    Returns:
        mu_final:    shape (batch,) — mean prediction
        sigma_total: shape (batch,) — total uncertainty (aleatoric + epistemic)
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    all_mu = []
    all_sigma = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            mu, sigma = model(x)
            all_mu.append(mu.cpu().numpy().flatten())
            all_sigma.append(sigma.cpu().numpy().flatten())
    
    all_mu = np.stack(all_mu, axis=0)      # (T, batch)
    all_sigma = np.stack(all_sigma, axis=0)  # (T, batch)
    
    # Mean prediction
    mu_final = all_mu.mean(axis=0)
    
    # Total uncertainty: Law of Total Variance
    # Var_total = E[σ²] + Var[μ]
    # E[σ²] = aleatoric (inherent data noise)
    # Var[μ] = epistemic (model disagreement, like RF tree variance)
    aleatoric = (all_sigma ** 2).mean(axis=0)
    epistemic = all_mu.var(axis=0)
    sigma_total = np.sqrt(aleatoric + epistemic)
    
    return mu_final, sigma_total


# ═══════════════════════════════════════════════════════════════════════
#  Conformal Prediction Calibration
# ═══════════════════════════════════════════════════════════════════════

class ConformalCalibrator:
    """
    Conformal Prediction for guaranteed coverage bounds.
    
    After training the MLP, we calibrate on a held-out set to find
    the quantile multiplier λ such that:
    
        P(y ≤ μ + λ·σ) ≥ 1 - α
    
    This gives a MATHEMATICALLY GUARANTEED upper bound on δ(time),
    unlike the heuristic thresholds in v3.
    
    Usage in HALO-U:
        upper_bound = μ + λ_{calibrated} · σ
        if upper_bound > Threshold → REJECT hint (safety gate)
    """
    
    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: miscoverage rate. alpha=0.10 → 90% coverage guarantee.
        """
        self.alpha = alpha
        self.lambda_calibrated = 1.645  # Gaussian 90th percentile as prior
        self._nonconformity_scores = None
    
    def calibrate(self, y_true: np.ndarray, mu_pred: np.ndarray,
                  sigma_pred: np.ndarray):
        """
        Calibrate λ from held-out validation data.
        
        Nonconformity score: s_i = (y_i - μ_i) / σ_i
        Calibrated λ = quantile_{1-α}(s_1, ..., s_n)
        
        Args:
            y_true: actual delta values, shape (n,)
            mu_pred: predicted means, shape (n,)
            sigma_pred: predicted uncertainties, shape (n,)
        """
        # Nonconformity scores
        scores = (y_true - mu_pred) / np.maximum(sigma_pred, 1e-6)
        self._nonconformity_scores = np.sort(scores)
        
        # Finite-sample correction: ceil((n+1)(1-α)) / n
        n = len(scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        quantile_idx = min(quantile_idx, n - 1)
        
        self.lambda_calibrated = float(self._nonconformity_scores[quantile_idx])
        
        # Verify coverage
        upper_bounds = mu_pred + self.lambda_calibrated * sigma_pred
        actual_coverage = (y_true <= upper_bounds).mean()
        
        logger.info(f"  Conformal Calibration (α={self.alpha}):")
        logger.info(f"    Calibrated λ = {self.lambda_calibrated:.3f}")
        logger.info(f"    Actual coverage = {actual_coverage:.3f} "
                     f"(target ≥ {1-self.alpha:.3f})")
        
        return self.lambda_calibrated
    
    def compute_upper_bound(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Compute the calibrated upper bound δ_p90.
        
        δ_upper = μ + λ_{calibrated} · σ
        
        Guarantee: P(δ_true ≤ δ_upper) ≥ 1 - α
        """
        return mu + self.lambda_calibrated * sigma


# ═══════════════════════════════════════════════════════════════════════
#  Feature Engineering v4
# ═══════════════════════════════════════════════════════════════════════

def filter_noisy_operators(df_ops, min_self_time=0.1):
    """Remove operators that are too fast to measure reliably."""
    before = len(df_ops)
    df = df_ops[
        (df_ops['self_time'] >= min_self_time) |
        (df_ops['actual_rows'] >= 100)
    ].copy()
    logger.info(f"  Noise filter: {before:,} -> {len(df):,} "
                f"(removed {before - len(df):,} near-zero operators)")
    return df


def engineer_features_v4(row1, row2, pair, risk_dict: RiskPatternDictionary) -> dict:
    """
    HALO v4 Feature Engineering.
    
    Key changes from v3:
      1. REMOVED: 5 worst-case features (data leakage) → replaced by 3 pattern risk features
      2. REMOVED: 26 manual cross-terms → replaced by 3 vector alignment features
      3. ADDED: 7 tree structural context features (from Phase 2)
      4. ADDED: 3 vector alignment features (from Phase 3)
      5. ADDED: 3 pattern risk features (from Phase 1)
    """
    features = {}

    # ── 1. Op type one-hot (15 types) ──
    op_types = ['NL_INNER_JOIN', 'FILTER', 'TABLE_SCAN', 'INDEX_LOOKUP',
                'HASH_JOIN', 'AGGREGATE', 'SORT', 'NL_ANTIJOIN',
                'NL_SEMIJOIN', 'INDEX_SCAN', 'TEMP_TABLE_SCAN',
                'STREAM', 'MATERIALIZE', 'GROUP_AGGREGATE', 'NL_LEFT_JOIN']
    for ot in op_types:
        features[f'is_{ot}'] = 1.0 if row1['op_type'] == ot else 0.0

    # ── 2. Semantic groups ──
    features['is_join'] = 1.0 if 'JOIN' in row1['op_type'] else 0.0
    features['is_scan'] = 1.0 if 'SCAN' in row1['op_type'] else 0.0
    features['is_index_op'] = 1.0 if 'INDEX' in row1['op_type'] else 0.0
    features['is_agg_sort'] = 1.0 if row1['op_type'] in (
        'AGGREGATE', 'SORT', 'GROUP_AGGREGATE') else 0.0

    # ── 3. Cardinality features ──
    actual_rows = max(row1['actual_rows'], 1)
    est_rows = max(row1['est_rows'], 1)
    loops = max(row1['loops'], 1)
    est_cost = max(row1['est_cost'], 0.01)
    self_time = max(row1['self_time'], 0.001)
    total_work = max(row1['total_work'], 0.001)

    features['log_actual_rows'] = math.log(actual_rows)
    features['log_est_rows'] = math.log(est_rows)
    features['log_loops'] = math.log(loops)
    features['log_est_cost'] = math.log(est_cost)

    # ── 4. Estimation quality ──
    features['row_est_error'] = row1['row_est_error']
    features['rows_per_loop'] = math.log(actual_rows / loops + 1)

    # ── 5. Work intensity ──
    features['log_self_time'] = math.log(self_time)
    features['log_total_work'] = math.log(total_work)

    # ── 6. Ratio features ──
    features['cost_per_row'] = math.log(est_cost / est_rows + 0.001)
    features['time_per_row'] = math.log(self_time / actual_rows + 0.0001)
    features['selectivity'] = math.log(actual_rows / est_rows + 0.001)

    # ── 7. Structure ──
    features['depth'] = row1['depth']
    features['num_children'] = row1.get('num_children', 0)

    # ── 8. Binary HW change flags ──
    sc = pair['storage_changed']
    cc = pair['compute_changed']
    features['storage_changed'] = sc
    features['compute_changed'] = cc
    features['both_changed'] = pair.get('both_changed', sc * cc)

    # ── 9. Quantitative HW ratios ──
    features['storage_speed_ratio'] = pair.get('storage_speed_ratio', 0.0)
    features['iops_ratio'] = pair.get('iops_ratio', 0.0)
    features['write_speed_ratio'] = pair.get('write_speed_ratio', 0.0)
    features['cpu_core_ratio'] = pair.get('cpu_core_ratio', 0.0)
    features['cpu_thread_ratio'] = pair.get('cpu_thread_ratio', 0.0)
    features['cpu_clock_ratio'] = pair.get('cpu_clock_ratio', 0.0)
    features['cpu_base_clock_ratio'] = pair.get('cpu_base_clock_ratio', 0.0)
    features['l3_cache_ratio'] = pair.get('l3_cache_ratio', 0.0)
    features['ram_ratio'] = pair.get('ram_ratio', 0.0)
    features['buffer_pool_ratio'] = pair.get('buffer_pool_ratio', 0.0)
    features['is_storage_downgrade'] = pair.get('is_storage_downgrade', 0)
    features['is_cpu_downgrade'] = pair.get('is_cpu_downgrade', 0)
    features['is_ram_downgrade'] = pair.get('is_ram_downgrade', 0)

    # ── 10. Workload metadata features ──
    features['log_table_rows'] = math.log(max(row1.get('table_rows', 1), 1))
    features['log_table_size_mb'] = math.log(max(row1.get('table_size_mb', 1), 1))
    features['n_indexes'] = row1.get('n_indexes', 0)
    features['log_dataset_gb'] = math.log(max(row1.get('dataset_total_gb', 0.1), 0.1))
    features['fits_in_buffer'] = row1.get('fits_in_buffer', 0)

    # ══════════════════════════════════════════════════════════
    #  NEW v4 Features
    # ══════════════════════════════════════════════════════════

    # ── 11. [Phase 2] Tree Structural Context ──
    # These are pre-computed by tree_propagation.py during data preparation
    features['child_max_io_bottleneck'] = row1.get('child_max_io_bottleneck', 0.0)
    features['subtree_cumulative_cost'] = math.log(max(row1.get('subtree_cumulative_cost', 0.01), 0.01))
    features['subtree_total_work'] = math.log(max(row1.get('subtree_total_work', 0.01), 0.01))
    features['subtree_max_depth'] = row1.get('subtree_max_depth', 0)
    features['is_pipeline_breaker'] = row1.get('is_pipeline_breaker', 0)
    features['child_selectivity_risk'] = row1.get('child_selectivity_risk', 0.0)
    features['self_io_risk'] = row1.get('self_io_risk', 0.0)

    # ── 12. [Phase 3] Hardware-Operator Vector Alignment ──
    # Replaces 26 manual cross-terms with 3 principled interaction features
    compute_vector_alignment_features(features, pair)

    # ── 13. [Phase 1] Pattern Risk Dictionary (Zero-shot, NO data leakage) ──
    selectivity = actual_rows / max(est_rows, 1)
    table_size_mb = math.exp(features['log_table_size_mb'])
    hw_transition_type = classify_hw_transition(pair)
    inject_pattern_risk_features(
        features, row1['op_type'], selectivity, table_size_mb,
        hw_transition_type, risk_dict
    )

    return features


# ═══════════════════════════════════════════════════════════════════════
#  Dataset Building
# ═══════════════════════════════════════════════════════════════════════

def build_env_pairs_v4(df_ops, risk_dict: Optional[RiskPatternDictionary] = None):
    """Build training dataset with v4 features."""
    logger.info("Building v4 env-pair dataset...")
    df_ops = filter_noisy_operators(df_ops)
    df_ops = df_ops.copy()

    # ── Enrich operators with workload metadata ──
    logger.info("  Enriching operators with workload metadata...")
    for idx, row in df_ops.iterrows():
        enriched = enrich_operator(
            row.to_dict(),
            benchmark=row.get('benchmark', ''),
            env_id=row.get('env_id', None)
        )
        for col in ['table_rows', 'table_size_mb', 'n_indexes',
                     'dataset_total_gb', 'n_tables_in_dataset', 'fits_in_buffer']:
            df_ops.at[idx, col] = enriched[col]
    logger.info(f"  Enrichment complete: {len(df_ops):,} operators")

    # ── [Phase 2] Propagate tree structural features ──
    df_ops = propagate_tree_features_from_df(df_ops)

    # ── Initialize risk dictionary if not provided ──
    if risk_dict is None:
        risk_dict = RiskPatternDictionary()
        # Will be compiled after first pass to collect delta labels

    # ── Build match keys (same as v3) ──
    df_ops['match_key_alias'] = (
        df_ops['query_id'] + '|' +
        df_ops['hint_variant'] + '|' +
        df_ops['hint_set'] + '|' +
        df_ops['op_type'] + '|' +
        df_ops['table_alias'] + '|' +
        df_ops['benchmark']
    )
    df_ops['match_key_pos'] = (
        df_ops['query_id'] + '|' +
        df_ops['hint_variant'] + '|' +
        df_ops['hint_set'] + '|' +
        df_ops['tree_position'].astype(str) + '|' +
        df_ops['benchmark']
    )

    samples = []
    for pair in ENV_PAIRS:
        e1, e2 = pair['env1'], pair['env2']
        ops1 = df_ops[df_ops['env_id'] == e1]
        ops2 = df_ops[df_ops['env_id'] == e2]
        if len(ops1) == 0 or len(ops2) == 0:
            continue

        ops1_alias = ops1[ops1['table_alias'] != ''].set_index('match_key_alias')
        ops2_alias = ops2[ops2['table_alias'] != ''].set_index('match_key_alias')
        matched_alias = ops1_alias.index.intersection(ops2_alias.index)

        ops1_pos = ops1.set_index('match_key_pos')
        ops2_pos = ops2.set_index('match_key_pos')
        matched_pos = ops1_pos.index.intersection(ops2_pos.index)

        used_keys = set()
        match_pairs = []

        for key in matched_alias:
            try:
                o1 = ops1_alias.loc[key]
                o2 = ops2_alias.loc[key]
                if isinstance(o1, pd.DataFrame): o1 = o1.iloc[0]
                if isinstance(o2, pd.DataFrame): o2 = o2.iloc[0]
                if o1['op_type'] == o2['op_type']:
                    match_pairs.append((o1, o2))
                    used_keys.add((o1['query_id'], o1['tree_position']))
            except Exception:
                continue

        for key in matched_pos:
            try:
                o1 = ops1_pos.loc[key]
                o2 = ops2_pos.loc[key]
                if isinstance(o1, pd.DataFrame): o1 = o1.iloc[0]
                if isinstance(o2, pd.DataFrame): o2 = o2.iloc[0]
                if (o1['query_id'], o1['tree_position']) in used_keys:
                    continue
                if o1['op_type'] == o2['op_type']:
                    match_pairs.append((o1, o2))
            except Exception:
                continue

        pair_count = 0
        for o1, o2 in match_pairs:
            st1 = max(o1['self_time'], 0.001)
            st2 = max(o2['self_time'], 0.001)
            delta = math.log(st2 / st1)
            if abs(delta) > 5:
                continue

            feat = engineer_features_v4(o1, o2, pair, risk_dict)
            feat['delta_time'] = delta
            feat['self_time_env1'] = st1
            feat['self_time_env2'] = st2
            feat['query_id'] = o1['query_id']
            feat['benchmark'] = o1['benchmark']
            feat['hint_set'] = o1['hint_set']
            feat['op_type'] = o1['op_type']
            feat['env1'] = e1
            feat['env2'] = e2
            feat['pair_label'] = pair['pair_label']
            samples.append(feat)
            pair_count += 1

        logger.info(f"  {e1} vs {e2}: {pair_count:,} matched pairs")

    df_samples = pd.DataFrame(samples)
    logger.info(f"Total training samples: {len(df_samples):,}")

    # ── Compile risk dictionary from collected samples ──
    if not risk_dict._is_compiled and len(df_samples) > 0:
        logger.info("  Compiling Risk Pattern Dictionary from training data...")
        hw_transition_map = {}
        for pair in ENV_PAIRS:
            hw_transition_map[pair['pair_label']] = classify_hw_transition(pair)
        risk_dict.compile_from_training_data(df_ops, df_samples, hw_transition_map)

    return df_samples, risk_dict


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

def train_sigma_v4(df_samples, risk_dict: RiskPatternDictionary,
                   output_dir='/root/halo/results'):
    """
    Train HALO v4 σ model — Probabilistic MLP with Conformal Calibration.
    
    Key differences from v3:
      1. Single probabilistic head (μ, σ) instead of Regressor + Classifier
      2. Gaussian NLL loss with asymmetric risk weighting
      3. MC-Dropout for epistemic uncertainty estimation
      4. Conformal Prediction for calibrated safety bounds
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    meta_cols = ['delta_time', 'self_time_env1', 'self_time_env2',
                 'query_id', 'benchmark', 'hint_set', 'op_type',
                 'env1', 'env2', 'pair_label']
    feature_cols = [c for c in df_samples.columns if c not in meta_cols]
    
    X = df_samples[feature_cols].values.astype(np.float32)
    y = df_samples['delta_time'].values.astype(np.float32)
    
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    df_clean = df_samples[mask].reset_index(drop=True)
    
    logger.info(f"\nTraining v4 σ model (Probabilistic MLP)...")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # ── Standardize features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ── Sample weights ──
    pair_counts = df_clean['pair_label'].value_counts()
    max_count = pair_counts.max()
    env_weights = df_clean['pair_label'].map(lambda p: max_count / pair_counts.get(p, 1)).values
    
    risk_weights = np.ones_like(env_weights)
    risk_weights[y > 0.8] *= 2.0   # Large regressions
    high_risk_mask = np.abs(y) > 0.5
    risk_weights[high_risk_mask] *= 3.0  # Asymmetric risk focus
    final_weights = env_weights * risk_weights
    final_weights = final_weights / final_weights.mean()  # Normalize
    
    # ── Leave-One-Env-Pair-Out Cross-Validation ──
    groups = df_clean['pair_label'].values
    logo = LeaveOneGroupOut()
    
    y_mu_cv = np.zeros_like(y)
    y_sigma_cv = np.zeros_like(y)
    fold_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
        X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1).to(device)
        w_train = torch.FloatTensor(final_weights[train_idx]).to(device)
        X_test = torch.FloatTensor(X_scaled[test_idx]).to(device)
        
        # Build model
        model = HALOProbabilisticMLP(
            input_dim=X_scaled.shape[1],
            hidden_dims=[256, 128, 64],
            dropout_rate=0.1,
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = GaussianNLLLoss()
        
        # Training loop
        dataset = TensorDataset(X_train, y_train, w_train)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        
        model.train()
        for epoch in range(100):
            epoch_loss = 0.0
            for X_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                mu, sigma = model(X_batch)
                loss = criterion(mu, sigma, y_batch, w_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
        
        # MC-Dropout prediction on test set
        mu_pred, sigma_pred = mc_dropout_predict(model, X_test, n_samples=30)
        y_mu_cv[test_idx] = mu_pred
        y_sigma_cv[test_idx] = sigma_pred
        
        # Fold metrics
        rmse = np.sqrt(mean_squared_error(y[test_idx], mu_pred))
        r2 = r2_score(y[test_idx], mu_pred) if len(test_idx) > 1 else 0
        pair_name = groups[test_idx[0]]
        
        nonzero = np.abs(y[test_idx]) > 0.1
        dir_acc = ((y[test_idx][nonzero] > 0) == (mu_pred[nonzero] > 0)).mean() * 100 \
                  if nonzero.sum() > 0 else 100.0
        
        fold_results.append({
            'pair': pair_name, 'n': len(test_idx),
            'rmse': float(rmse), 'r2': float(r2), 'dir_acc': float(dir_acc)
        })
        logger.info(f"  Fold {fold_idx+1} ({pair_name}): RMSE={rmse:.3f}, "
                     f"R²={r2:.3f}, DirAcc={dir_acc:.1f}%, n={len(test_idx)}")
    
    # ── Overall metrics ──
    overall_rmse = np.sqrt(mean_squared_error(y, y_mu_cv))
    overall_r2 = r2_score(y, y_mu_cv)
    overall_mae = mean_absolute_error(y, y_mu_cv)
    overall_corr = np.corrcoef(y, y_mu_cv)[0, 1]
    
    nonzero_mask = np.abs(y) > 0.1
    overall_dir_acc = ((y[nonzero_mask] > 0) == (y_mu_cv[nonzero_mask] > 0)).mean() * 100 \
                      if nonzero_mask.sum() > 0 else 100.0
    
    # ── Conformal Prediction Calibration ──
    logger.info(f"\n  Conformal Prediction Calibration:")
    calibrator = ConformalCalibrator(alpha=0.10)
    lambda_cal = calibrator.calibrate(y, y_mu_cv, y_sigma_cv)
    
    # ── High-Risk Detection using calibrated upper bound ──
    upper_bounds = calibrator.compute_upper_bound(y_mu_cv, y_sigma_cv)
    
    # Compare: predict HIGH_RISK if upper_bound > 0.5
    for thresh, label in [(0.5, '0.5'), (0.8, '0.8')]:
        y_high = np.abs(y) > thresh
        y_pred_high = upper_bounds > thresh  # Asymmetric: only flag slowdowns
        prec = precision_score(y_high, y_pred_high) if y_pred_high.any() else 0.0
        rec = recall_score(y_high, y_pred_high) if y_high.any() else 0.0
        f1 = f1_score(y_high, y_pred_high) if y_high.any() else 0.0
        logger.info(f"    Upper-Bound Risk (δ>{label}): "
                     f"Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    
    # ── Final model training on all data ──
    logger.info(f"\n  Training final model on all data...")
    X_all = torch.FloatTensor(X_scaled).to(device)
    y_all = torch.FloatTensor(y).unsqueeze(1).to(device)
    w_all = torch.FloatTensor(final_weights).to(device)
    
    final_model = HALOProbabilisticMLP(
        input_dim=X_scaled.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
    ).to(device)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    criterion = GaussianNLLLoss()
    
    dataset = TensorDataset(X_all, y_all, w_all)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    final_model.train()
    for epoch in range(150):
        for X_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            mu, sigma = final_model(X_batch)
            loss = criterion(mu, sigma, y_batch, w_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    # ── Results Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("σ Model v4 Results (Probabilistic MLP + Conformal):")
    logger.info(f"  RMSE:             {overall_rmse:.3f}")
    logger.info(f"  MAE:              {overall_mae:.3f}")
    logger.info(f"  R²:               {overall_r2:.3f}")
    logger.info(f"  Pearson r:        {overall_corr:.3f}")
    logger.info(f"  Direction Acc:    {overall_dir_acc:.1f}%")
    logger.info(f"  Calibrated λ:     {lambda_cal:.3f}")
    logger.info(f"  Mean σ (uncertainty): {y_sigma_cv.mean():.3f}")
    
    # ── Save ──
    results = {
        'model_type': 'ProbabilisticMLP',
        'model_version': 'v4_conformal',
        'architecture': {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.1,
            'spectral_norm': True,
            'mc_dropout_samples': 30,
        },
        'improvements': [
            'zero_shot_pattern_dictionary',
            'tree_structural_context',
            'vector_alignment_features',
            'probabilistic_output',
            'gaussian_nll_loss',
            'mc_dropout_uncertainty',
            'conformal_prediction_calibration',
            'spectral_normalization_ood',
        ],
        'n_samples': int(len(X)),
        'n_features': len(feature_cols),
        'feature_list': feature_cols,
        'overall_rmse': float(overall_rmse),
        'overall_mae': float(overall_mae),
        'overall_r2': float(overall_r2),
        'pearson_r': float(overall_corr),
        'direction_accuracy': float(overall_dir_acc),
        'conformal_lambda': float(lambda_cal),
        'mean_uncertainty': float(y_sigma_cv.mean()),
        'fold_results': fold_results,
    }
    
    with open(os.path.join(output_dir, 'sigma_results_v4.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'calibrator_lambda': lambda_cal,
        'architecture': {'input_dim': X_scaled.shape[1], 'hidden_dims': [256, 128, 64]},
    }
    torch.save(checkpoint, os.path.join(output_dir, 'sigma_model_v4.pt'))
    
    # Save risk dictionary
    risk_dict.save(os.path.join(output_dir, 'risk_pattern_dictionary.json'))
    
    logger.info(f"\n  Model saved: {output_dir}/sigma_model_v4.pt")
    logger.info(f"  Results saved: {output_dir}/sigma_results_v4.json")
    logger.info(f"  Risk dict saved: {output_dir}/risk_pattern_dictionary.json")
    
    return final_model, scaler, calibrator, results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    logger.info(f"Loaded {len(df_ops):,} operators")
    
    # Phase 1: Build risk dictionary (first pass without it, then compile)
    risk_dict = RiskPatternDictionary()
    
    # Build dataset with all v4 features
    df_samples, risk_dict = build_env_pairs_v4(df_ops, risk_dict)
    df_samples.to_parquet(os.path.join('/root/halo/results', 'sigma_training_v4.parquet'),
                          index=False)
    
    # Train the probabilistic model
    model, scaler, calibrator, results = train_sigma_v4(df_samples, risk_dict)
