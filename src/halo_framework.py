# -*- coding: utf-8 -*-
"""
HALO Framework — Hardware-Aware Learned Optimizer
==================================================

A unified hint recommendation framework with two selection policies:

  HALO-G (Greedy):
      Experience-based policy. Picks the hint that performed best in the
      source environment and applies it directly to the target. Fast and
      aggressive — maximizes expected speedup but ignores hardware
      differences, so it can cause regressions on new hardware.

  HALO-R (Robust):
      σ-aware policy. Uses a learned operator-level hardware sensitivity
      model (σ model) to predict which operators will behave differently
      on the target hardware. Rejects hints whose affected operators carry
      unacceptable risk. Sacrifices a small amount of peak speedup for
      drastically fewer regressions.

Pipeline:
  1. PROFILING   — Collect EXPLAIN ANALYZE data across environments
  2. TRAINING    — Build operator-level σ (sensitivity) model
  3. RECOMMEND   — For any query, recommend the safest best hint

Usage:
    halo = HaloFramework()
    halo.train(operators_df, queries_df)

    # Greedy mode (experience-only, like BAO)
    rec_g = halo.recommend('tpch_q9', 'A_NVMe', 'B_SATA', policy='greedy')

    # Robust mode (σ-aware, our key contribution)
    rec_r = halo.recommend('tpch_q9', 'A_NVMe', 'B_SATA', policy='robust')

    # Compare both
    halo.compare_policies('A_NVMe', 'B_SATA')
"""

import pandas as pd
import numpy as np
import math
import pickle
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('halo')


from hw_registry import HW_REGISTRY, compute_hw_features
from workload_metadata import enrich_operator

# Backward compatibility
ENV_PROFILES = HW_REGISTRY


# ═══════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OperatorRisk:
    """Risk assessment for a single operator under a hint."""
    op_type: str
    table_alias: str
    depth: int
    predicted_delta: float      # log(target_time / source_time)
    risk_level: str             # 'SAFE', 'MODERATE', 'HIGH'
    contribution: str           # 'BENEFIT' or 'HURT'


@dataclass
class HintCandidate:
    """Evaluation of one hint set for a query."""
    hint_set: str
    source_speedup: float
    predicted_target_speedup: float
    n_operators_affected: int
    n_high_risk: int
    n_benefit: int
    total_risk_score: float
    operator_details: List[OperatorRisk] = field(default_factory=list)
    expected_gain: float = 1.0          # Risk-Reward expected value
    risk_probability: float = 0.0       # Estimated regression probability


@dataclass
class HaloRecommendation:
    """Final recommendation from the HALO framework."""
    query_id: str
    source_env: str
    target_env: str
    policy: str                       # 'greedy' or 'robust'
    recommended_hint: Optional[str]   # None = keep native
    predicted_speedup: float
    risk_level: str
    reason: str
    all_candidates: List[HintCandidate] = field(default_factory=list)
    native_fallback: bool = False


# ═══════════════════════════════════════════════════════════════════════
#  HALO Framework Core
# ═══════════════════════════════════════════════════════════════════════

class HaloFramework:
    """
    Hardware-Aware Learned Optimizer.

    Provides two hint selection policies under a single unified API:

      policy='greedy'  →  HALO-G  (experience-based, aggressive)
      policy='robust'  →  HALO-R  (σ-aware, safe)
    """

    # ── Configurable thresholds for HALO-R ──
    RISK_HIGH_THRESHOLD = 0.8    # Default; overridden by adaptive thresholds
    RISK_MODERATE_THRESHOLD = 0.4
    MIN_SPEEDUP_THRESHOLD = 1.05
    MAX_HIGH_RISK_OPS = 3        # Increased from 0 to allow minor uncertainties

    # ── Adaptive per-operator risk thresholds ──
    # Operators with naturally high variance get relaxed thresholds;
    # stable operators get tighter thresholds.
    ADAPTIVE_RISK_THRESHOLDS = {
        # I/O-heavy ops: storage change causes large natural σ
        'TABLE_SCAN':       1.2,
        'INDEX_SCAN':       1.0,
        'INDEX_LOOKUP':     1.0,
        'TEMP_TABLE_SCAN':  1.0,
        # CPU-heavy ops: sensitive to compute change
        'HASH_JOIN':        0.7,
        'NL_INNER_JOIN':    0.7,
        'NL_SEMIJOIN':      0.7,
        'NL_ANTIJOIN':      0.7,
        'NL_LEFT_JOIN':     0.7,
        # Lightweight ops: even small delta is suspicious
        'FILTER':           0.5,
        'AGGREGATE':        0.5,
        'GROUP_AGGREGATE':  0.5,
        'SORT':             0.6,
        'STREAM':           0.5,
        'MATERIALIZE':      0.6,
    }

    # ── Risk-Reward tradeoff constants ──
    HIGH_CONFIDENCE_SPEEDUP = 2.0   # Source speedup above this = high-value hint
    RISK_REWARD_MIN_EV = 1.10       # Minimum expected value to accept risky hint

    # ── Threshold for HALO-G ──
    GREEDY_MIN_SPEEDUP = 1.02    # must beat native by 2% in source

    def __init__(self):
        self.sigma_model = None
        self.feature_cols = None
        self.df_operators = None
        self.df_queries = None
        self._operator_index = {}
        self._query_baselines = {}
        self._trained = False

    # ───────────────────────────────────────────────────────────────────
    #  Phase 1: Training / Initialization
    # ───────────────────────────────────────────────────────────────────

    def train(self, df_operators: pd.DataFrame, df_queries: pd.DataFrame,
              model_path: str = '/root/halo/results/sigma_model_v3.pkl'):
        """
        Load profiling data and the pre-trained σ model.

        Args:
            df_operators: unified_operators.parquet
            df_queries:   unified_queries.parquet
            model_path:   path to σ model pickle (from sigma_model_v3.py)
        """
        logger.info("═══ HALO Framework: Initializing ═══")

        self.df_operators = df_operators.copy()
        self.df_queries = df_queries.copy()

        # Enrich operators with workload metadata
        logger.info("  Enriching operators with workload metadata...")
        for idx, row in self.df_operators.iterrows():
            enriched = enrich_operator(
                row.to_dict(),
                benchmark=row.get('benchmark', ''),
                env_id=row.get('env_id', None)
            )
            for col in ['table_rows', 'table_size_mb', 'n_indexes',
                         'dataset_total_gb', 'n_tables_in_dataset', 'fits_in_buffer']:
                self.df_operators.at[idx, col] = enriched[col]

        # Load σ model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            self.sigma_model = saved['model']
            self.feature_cols = saved['feature_cols']
            logger.info(f"  σ model loaded: {len(self.feature_cols)} features")
        else:
            logger.warning(f"  σ model not found at {model_path}. "
                           "HALO-R will be unavailable. Run sigma_model_v3.py first.")

        self._build_operator_index()
        self._build_baseline_index()

        self._trained = True
        logger.info(f"  Operators indexed: {len(self._operator_index):,} groups")
        logger.info(f"  Query baselines:   {len(self._query_baselines):,}")
        logger.info("═══ HALO Framework: Ready ═══\n")

    def _build_operator_index(self):
        self._operator_index = {}
        for (env, qid, hs), group in self.df_operators.groupby(
                ['env_id', 'query_id', 'hint_set']):
            self._operator_index[(env, qid, hs)] = group.to_dict('records')

    def _build_baseline_index(self):
        self._query_baselines = {}
        baselines = self.df_queries[self.df_queries['hint_set'] == 'baseline']
        for _, row in baselines.iterrows():
            key = (row['env_id'], row['query_id'])
            t = row['execution_time_s']
            if key not in self._query_baselines or t < self._query_baselines[key]:
                self._query_baselines[key] = t

    # ───────────────────────────────────────────────────────────────────
    #  Phase 2: Operator-Level σ Analysis  (used by HALO-R only)
    # ───────────────────────────────────────────────────────────────────

    def _get_hw_change_flags(self, source_env: str, target_env: str) -> dict:
        """Compute quantitative HW transition features using the registry."""
        return compute_hw_features(source_env, target_env)

    def _engineer_features_for_prediction(self, op: dict, hw_change: dict) -> dict:
        features = {}
        op_types = ['NL_INNER_JOIN', 'FILTER', 'TABLE_SCAN', 'INDEX_LOOKUP',
                    'HASH_JOIN', 'AGGREGATE', 'SORT', 'NL_ANTIJOIN',
                    'NL_SEMIJOIN', 'INDEX_SCAN', 'TEMP_TABLE_SCAN',
                    'STREAM', 'MATERIALIZE', 'GROUP_AGGREGATE', 'NL_LEFT_JOIN']
        for ot in op_types:
            features[f'is_{ot}'] = 1.0 if op['op_type'] == ot else 0.0

        features['is_join'] = 1.0 if 'JOIN' in op['op_type'] else 0.0
        features['is_scan'] = 1.0 if 'SCAN' in op['op_type'] else 0.0
        features['is_index_op'] = 1.0 if 'INDEX' in op['op_type'] else 0.0
        features['is_agg_sort'] = 1.0 if op['op_type'] in (
            'AGGREGATE', 'SORT', 'GROUP_AGGREGATE') else 0.0

        features['log_actual_rows'] = math.log(max(op.get('actual_rows', 1), 1))
        features['log_est_rows'] = math.log(max(op.get('est_rows', 1), 1))
        features['log_loops'] = math.log(max(op.get('loops', 1), 1))
        features['log_est_cost'] = math.log(max(op.get('est_cost', 0.01), 0.01))

        features['row_est_error'] = op.get('row_est_error', 1.0)
        features['rows_per_loop'] = math.log(
            max(op.get('actual_rows', 1), 1) / max(op.get('loops', 1), 1) + 1)

        features['log_self_time'] = math.log(max(op.get('self_time', 0.001), 0.001))
        features['log_total_work'] = math.log(max(op.get('total_work', 0.001), 0.001))

        # Ratio features
        est_rows = max(op.get('est_rows', 1), 1)
        est_cost = max(op.get('est_cost', 0.01), 0.01)
        actual_rows = max(op.get('actual_rows', 1), 1)
        self_time = max(op.get('self_time', 0.001), 0.001)
        features['cost_per_row'] = math.log(est_cost / est_rows + 0.001)
        features['time_per_row'] = math.log(self_time / actual_rows + 0.0001)
        features['selectivity'] = math.log(actual_rows / est_rows + 0.001)

        features['depth'] = op.get('depth', 0)
        features['num_children'] = op.get('num_children', 0)

        # Binary HW flags
        sc = hw_change['storage_changed']
        cc = hw_change['compute_changed']
        features['storage_changed'] = sc
        features['compute_changed'] = cc
        features['both_changed'] = hw_change.get('both_changed', sc * cc)

        # Quantitative HW ratios
        features['storage_speed_ratio'] = hw_change.get('storage_speed_ratio', 0.0)
        features['iops_ratio'] = hw_change.get('iops_ratio', 0.0)
        features['write_speed_ratio'] = hw_change.get('write_speed_ratio', 0.0)
        features['cpu_core_ratio'] = hw_change.get('cpu_core_ratio', 0.0)
        features['cpu_thread_ratio'] = hw_change.get('cpu_thread_ratio', 0.0)
        features['cpu_clock_ratio'] = hw_change.get('cpu_clock_ratio', 0.0)
        features['cpu_base_clock_ratio'] = hw_change.get('cpu_base_clock_ratio', 0.0)
        features['l3_cache_ratio'] = hw_change.get('l3_cache_ratio', 0.0)
        features['ram_ratio'] = hw_change.get('ram_ratio', 0.0)
        features['buffer_pool_ratio'] = hw_change.get('buffer_pool_ratio', 0.0)
        features['is_storage_downgrade'] = hw_change.get('is_storage_downgrade', 0)
        features['is_cpu_downgrade'] = hw_change.get('is_cpu_downgrade', 0)
        features['is_ram_downgrade'] = hw_change.get('is_ram_downgrade', 0)

        # Workload metadata
        features['log_table_rows'] = math.log(max(op.get('table_rows', 1), 1))
        features['log_table_size_mb'] = math.log(max(op.get('table_size_mb', 1), 1))
        features['n_indexes'] = op.get('n_indexes', 0)
        features['log_dataset_gb'] = math.log(max(op.get('dataset_total_gb', 0.1), 0.1))
        features['fits_in_buffer'] = op.get('fits_in_buffer', 0)

        # Operator × Binary HW interaction
        features['scan_x_storage'] = features['is_scan'] * sc
        features['scan_x_compute'] = features['is_scan'] * cc
        features['join_x_storage'] = features['is_join'] * sc
        features['join_x_compute'] = features['is_join'] * cc
        features['index_x_storage'] = features['is_index_op'] * sc
        features['index_x_compute'] = features['is_index_op'] * cc
        features['agg_x_storage'] = features['is_agg_sort'] * sc
        features['agg_x_compute'] = features['is_agg_sort'] * cc

        # Work × Binary HW interaction
        features['work_x_storage'] = features['log_total_work'] * sc
        features['work_x_compute'] = features['log_total_work'] * cc
        features['rows_x_storage'] = features['log_actual_rows'] * sc
        features['rows_x_compute'] = features['log_actual_rows'] * cc

        # Operator × Quantitative HW interaction
        iops_r = hw_change.get('iops_ratio', 0.0)
        clock_r = hw_change.get('cpu_clock_ratio', 0.0)
        core_r = hw_change.get('cpu_core_ratio', 0.0)
        features['scan_x_iops'] = features['is_scan'] * iops_r
        features['scan_x_storage_speed'] = features['is_scan'] * hw_change.get('storage_speed_ratio', 0.0)
        features['join_x_cpu_clock'] = features['is_join'] * clock_r
        features['join_x_cpu_cores'] = features['is_join'] * core_r
        features['agg_x_cpu_clock'] = features['is_agg_sort'] * clock_r
        features['index_x_iops'] = features['is_index_op'] * iops_r

        # Work × Quantitative HW interaction
        features['rows_x_iops'] = features['log_actual_rows'] * iops_r
        features['work_x_cpu_clock'] = features['log_total_work'] * clock_r
        features['cost_x_iops'] = features['log_est_cost'] * iops_r

        # Workload × HW cross-terms
        features['table_size_x_iops'] = features['log_table_size_mb'] * iops_r
        features['table_size_x_storage_speed'] = features['log_table_size_mb'] * hw_change.get('storage_speed_ratio', 0.0)
        features['table_rows_x_iops'] = features['log_table_rows'] * iops_r
        features['buffer_miss_x_iops'] = (1 - features['fits_in_buffer']) * iops_r
        features['dataset_x_ram'] = features['log_dataset_gb'] * hw_change.get('ram_ratio', 0.0)

        return features

    def _predict_operator_delta(self, op: dict, hw_change: dict) -> float:
        features = self._engineer_features_for_prediction(op, hw_change)
        feat_vector = np.array(
            [features.get(col, 0.0) for col in self.feature_cols],
            dtype=np.float32
        ).reshape(1, -1)
        return float(self.sigma_model.predict(feat_vector)[0])

    def _diff_operators(self, baseline_ops: List[dict],
                        hinted_ops: List[dict]) -> List[dict]:
        bl_by_pos = {op['tree_position']: op for op in baseline_ops}
        ht_by_pos = {op['tree_position']: op for op in hinted_ops}
        changed = []
        for pos in set(bl_by_pos) | set(ht_by_pos):
            bl_op = bl_by_pos.get(pos)
            ht_op = ht_by_pos.get(pos)
            if bl_op and ht_op:
                if bl_op['op_type'] != ht_op['op_type']:
                    changed.append({'type': 'method_change', 'operator': ht_op})
                elif abs(bl_op.get('est_cost', 0) - ht_op.get('est_cost', 0)) > 0.01:
                    changed.append({'type': 'cost_change', 'operator': ht_op})
            elif ht_op:
                changed.append({'type': 'added', 'operator': ht_op})
            elif bl_op:
                changed.append({'type': 'removed', 'operator': bl_op})
        return changed

    def _get_adaptive_threshold(self, op_type: str) -> float:
        """Get per-operator risk threshold. Operators with naturally
        high variance (I/O ops) get relaxed thresholds; stable ops
        (FILTER, AGGREGATE) get tighter ones."""
        return self.ADAPTIVE_RISK_THRESHOLDS.get(
            op_type, self.RISK_HIGH_THRESHOLD
        )

    def _assess_hint_risk(self, query_id: str, hint_set: str,
                          source_env: str, target_env: str) -> Optional[HintCandidate]:
        """Core HALO logic: assess operator-level risk of a hint on target HW.
        
        Improvements over baseline:
          1. Adaptive per-operator thresholds (I/O ops tolerate higher σ)
          2. Risk-Reward expected value calculation
          3. Confidence tier assignment for downstream policy
        """
        hw_change = self._get_hw_change_flags(source_env, target_env)

        baseline_ops = self._operator_index.get((source_env, query_id, 'baseline'), [])
        hinted_ops = self._operator_index.get((source_env, query_id, hint_set), [])
        if not baseline_ops or not hinted_ops:
            return None

        baseline_time = self._query_baselines.get((source_env, query_id))
        hinted_times = self.df_queries[
            (self.df_queries['env_id'] == source_env) &
            (self.df_queries['query_id'] == query_id) &
            (self.df_queries['hint_set'] == hint_set)
        ]['execution_time_s']
        if baseline_time is None or len(hinted_times) == 0:
            return None

        best_hinted_time = hinted_times.min()
        source_speedup = baseline_time / max(best_hinted_time, 0.001)

        diffs = self._diff_operators(baseline_ops, hinted_ops)

        op_risks = []
        total_risk_score = 0.0
        n_high = 0
        n_benefit = 0
        for diff in diffs:
            op = diff['operator']
            
            # IDENTITY BYPASS: If HW is identical, there is NO hardware-driven risk.
            if not hw_change['compute_changed'] and not hw_change['storage_changed']:
                delta = 0.0
            else:
                delta = self._predict_operator_delta(op, hw_change)
            
            # ── Adaptive threshold per operator type ──
            op_threshold = self._get_adaptive_threshold(op['op_type'])
            op_moderate_threshold = op_threshold * 0.5
            
            if delta > op_threshold:
                risk_level = 'HIGH'; n_high += 1
            elif delta > op_moderate_threshold:
                risk_level = 'MODERATE'
            elif delta < -op_threshold:
                risk_level = 'SAFE'  # High confidence improvement
            else:
                risk_level = 'SAFE'
                
            contribution = 'HURT' if delta > 0 else 'BENEFIT'
            if delta <= 0: n_benefit += 1
            weight = math.log(max(op.get('self_time', 0.001), 0.001) + 1) + 1
            total_risk_score += max(0, delta) * weight  # Only penalize slowdowns
            op_risks.append(OperatorRisk(
                op_type=op['op_type'],
                table_alias=op.get('table_alias', ''),
                depth=op.get('depth', 0),
                predicted_delta=delta,
                risk_level=risk_level,
                contribution=contribution,
            ))

        avg_delta = np.mean([r.predicted_delta for r in op_risks]) if op_risks else 0.0
        adjustment_factor = math.exp(-avg_delta)
        predicted_target_speedup = 1.0 + (source_speedup - 1.0) * adjustment_factor

        # ── Risk-Reward Expected Value ──
        # Estimate probability of regression based on risk score
        risk_probability = min(1.0, total_risk_score / max(len(diffs), 1))
        expected_gain = source_speedup * (1.0 - risk_probability)

        return HintCandidate(
            hint_set=hint_set,
            source_speedup=source_speedup,
            predicted_target_speedup=predicted_target_speedup,
            n_operators_affected=len(diffs),
            n_high_risk=n_high,
            n_benefit=n_benefit,
            total_risk_score=total_risk_score,
            operator_details=op_risks,
            expected_gain=expected_gain,
            risk_probability=risk_probability,
        )

    # ───────────────────────────────────────────────────────────────────
    #  Phase 3: Recommendation  (Unified API)
    # ───────────────────────────────────────────────────────────────────

    def recommend(self, query_id: str, source_env: str,
                  target_env: str, policy: str = 'robust') -> HaloRecommendation:
        """
        Recommend the best hint for a query on target hardware.

        Args:
            query_id:   e.g. 'tpch_q9', '1a'
            source_env: environment where hints were profiled ('A_NVMe')
            target_env: target deployment environment ('B_SATA')
            policy:     'greedy' for HALO-G, 'robust' for HALO-R

        Returns:
            HaloRecommendation
        """
        if not self._trained:
            raise RuntimeError("HALO not trained. Call halo.train() first.")
        if policy not in ('greedy', 'robust'):
            raise ValueError(f"Unknown policy '{policy}'. Use 'greedy' or 'robust'.")

        # Discover available hints
        available_hints = set()
        for (env, qid, hs) in self._operator_index:
            if env == source_env and qid == query_id and hs != 'baseline':
                available_hints.add(hs)

        if not available_hints:
            return HaloRecommendation(
                query_id=query_id, source_env=source_env, target_env=target_env,
                policy=policy, recommended_hint=None, predicted_speedup=1.0,
                risk_level='N/A',
                reason=f"No hint data for {query_id} in {source_env}",
                native_fallback=True,
            )

        # ────────────────────────────────────────────────────
        #  HALO-G: Greedy (experience-based)
        # ────────────────────────────────────────────────────
        if policy == 'greedy':
            return self._recommend_greedy(query_id, source_env, target_env,
                                          available_hints)

        # ────────────────────────────────────────────────────
        #  HALO-R: Robust (σ-aware)
        # ────────────────────────────────────────────────────
        return self._recommend_robust(query_id, source_env, target_env,
                                      available_hints)

    def _recommend_greedy(self, query_id, source_env, target_env,
                          available_hints) -> HaloRecommendation:
        """
        HALO-G: Pick the hint with the best speedup in the source env.
        No operator-level analysis — pure experience replay.
        """
        src_baseline = self._query_baselines.get((source_env, query_id), 1.0)

        best_hs = None
        best_speedup = 1.0

        for hs in available_hints:
            ht = self.df_queries[
                (self.df_queries['env_id'] == source_env) &
                (self.df_queries['query_id'] == query_id) &
                (self.df_queries['hint_set'] == hs)
            ]['execution_time_s']
            if len(ht) > 0:
                speedup = src_baseline / max(ht.min(), 0.001)
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_hs = hs

        if best_hs and best_speedup >= self.GREEDY_MIN_SPEEDUP:
            return HaloRecommendation(
                query_id=query_id, source_env=source_env, target_env=target_env,
                policy='greedy',
                recommended_hint=best_hs,
                predicted_speedup=best_speedup,  # assumes target ≈ source
                risk_level='UNKNOWN',
                reason=(f"HALO-G selected {best_hs} based on source speedup "
                        f"{best_speedup:.2f}x. No hardware risk analysis performed."),
            )
        else:
            return HaloRecommendation(
                query_id=query_id, source_env=source_env, target_env=target_env,
                policy='greedy',
                recommended_hint=None,
                predicted_speedup=1.0,
                risk_level='SAFE',
                reason="HALO-G: No hint beat native in source environment.",
                native_fallback=True,
            )

    def _recommend_robust(self, query_id, source_env, target_env,
                          available_hints) -> HaloRecommendation:
        """
        HALO-R (v2): 3-Tier Confidence + Risk-Reward Policy.

        Improvement over v1:
          - Tier 1 (GREEN):  High-confidence safe → always apply
          - Tier 2 (YELLOW): Uncertain → apply only if Risk-Reward EV > threshold
          - Tier 3 (RED):    High-risk → reject (native fallback)
        """
        if self.sigma_model is None:
            logger.warning("σ model not loaded. Falling back to HALO-G.")
            return self._recommend_greedy(query_id, source_env, target_env,
                                          available_hints)

        candidates = []
        for hs in sorted(available_hints):
            c = self._assess_hint_risk(query_id, hs, source_env, target_env)
            if c:
                candidates.append(c)

        if not candidates:
            return HaloRecommendation(
                query_id=query_id, source_env=source_env, target_env=target_env,
                policy='robust', recommended_hint=None, predicted_speedup=1.0,
                risk_level='N/A',
                reason="HALO-R: Could not evaluate any hint candidates.",
                native_fallback=True,
            )

        # ── 3-Tier Classification ──
        tier1_green = []   # High confidence: safe AND good speedup
        tier2_yellow = []  # Uncertain: some risk but potentially rewarding
        tier3_red = []     # Dangerous: too many high-risk ops

        for c in candidates:
            if c.n_high_risk == 0 and c.predicted_target_speedup >= self.MIN_SPEEDUP_THRESHOLD:
                # TIER 1 (GREEN): No high-risk ops, clear speedup
                tier1_green.append(c)
            elif c.n_high_risk > self.MAX_HIGH_RISK_OPS:
                # TIER 3 (RED): Too many dangerous operators
                tier3_red.append(c)
            else:
                # TIER 2 (YELLOW): Some risk, needs Risk-Reward evaluation
                tier2_yellow.append(c)

        # ── Selection Logic ──
        best = None
        tier_used = None

        # Priority 1: Pick from GREEN tier
        if tier1_green:
            best = max(tier1_green, key=lambda c: c.predicted_target_speedup)
            tier_used = 'GREEN'

        # Priority 2: Evaluate YELLOW tier via Risk-Reward EV
        if not best and tier2_yellow:
            # Accept yellow candidates only if expected gain justifies the risk
            rewarding = []
            for c in tier2_yellow:
                # High-value hints (source speedup > 2x) get special consideration
                if c.source_speedup >= self.HIGH_CONFIDENCE_SPEEDUP:
                    # For high-value hints: accept if EV still beats native
                    if c.expected_gain >= self.RISK_REWARD_MIN_EV:
                        rewarding.append(c)
                else:
                    # For moderate hints: stricter — need predicted target speedup
                    if (c.predicted_target_speedup >= self.MIN_SPEEDUP_THRESHOLD
                            and c.expected_gain >= self.RISK_REWARD_MIN_EV):
                        rewarding.append(c)

            if rewarding:
                best = max(rewarding, key=lambda c: c.expected_gain)
                tier_used = 'YELLOW'

        # Priority 3: Also check GREEN from initially-yellow if they pass
        #   (candidates that were borderline could still be safe)
        if not best and tier1_green:
            best = max(tier1_green, key=lambda c: c.predicted_target_speedup)
            tier_used = 'GREEN'

        if best:
            if best.total_risk_score < 1.0:
                risk = 'SAFE'
            elif best.total_risk_score < 3.0:
                risk = 'MODERATE'
            else:
                risk = 'HIGH'

            return HaloRecommendation(
                query_id=query_id, source_env=source_env, target_env=target_env,
                policy='robust',
                recommended_hint=best.hint_set,
                predicted_speedup=best.predicted_target_speedup,
                risk_level=risk,
                reason=(f"HALO-R [{tier_used}] selected {best.hint_set}: "
                        f"source {best.source_speedup:.2f}x → "
                        f"predicted target {best.predicted_target_speedup:.2f}x. "
                        f"{best.n_operators_affected} ops affected "
                        f"({best.n_benefit}↑, {best.n_high_risk}⚠). "
                        f"EV={best.expected_gain:.2f}, "
                        f"Risk={best.total_risk_score:.2f}"),
                all_candidates=candidates,
            )
        else:
            best_rej = max(candidates, key=lambda c: c.source_speedup)
            return HaloRecommendation(
                query_id=query_id, source_env=source_env, target_env=target_env,
                policy='robust',
                recommended_hint=None,
                predicted_speedup=1.0,
                risk_level='SAFE',
                reason=(f"HALO-R rejected all hints for safety. "
                        f"Best was {best_rej.hint_set} (src {best_rej.source_speedup:.2f}x, "
                        f"EV={best_rej.expected_gain:.2f}) "
                        f"but {best_rej.n_high_risk} high-risk ops detected. "
                        f"Keeping native to avoid regression."),
                all_candidates=candidates,
                native_fallback=True,
            )

    # ───────────────────────────────────────────────────────────────────
    #  Batch Operations
    # ───────────────────────────────────────────────────────────────────

    def _build_query_vectors(self, df_ops):
        """Build feature vectors for queries to find similarities."""
        logger.info("Building query similarity vectors...")
        vectors = []
        for (env, qid), group in df_ops[df_ops['hint_set'] == 'baseline'].groupby(['env_id', 'query_id']):
            vec = {'env_id': env, 'query_id': qid}
            # Count operator types
            counts = group['op_type'].value_counts()
            for ot, count in counts.items():
                vec[f'count_{ot}'] = count
            # Sum self times
            vec['total_self_time'] = group['self_time'].sum()
            vectors.append(vec)
        
        self._query_vectors = pd.DataFrame(vectors).fillna(0)

    def find_similar_queries(self, target_env, plan_df, top_n=3):
        """Find the most similar historical queries based on their baseline plans."""
        # Build vector for the new query plan
        new_vec = {}
        counts = plan_df['op_type'].value_counts()
        for ot, count in counts.items():
            new_vec[f'count_{ot}'] = count
        
        # Calculate similarity (simple Euclidean for now)
        hist_vecs = self._query_vectors[self._query_vectors['env_id'] == target_env].copy()
        if hist_vecs.empty: return []

        # Ensure same columns
        all_cols = [c for c in hist_vecs.columns if c.startswith('count_')]
        for c in all_cols:
            if c not in new_vec: new_vec[c] = 0
            
        distances = []
        new_v = np.array([new_vec.get(c, 0) for c in all_cols])
        
        for idx, row in hist_vecs.iterrows():
            row_v = np.array([row.get(c, 0) for c in all_cols])
            dist = np.linalg.norm(new_v - row_v)
            distances.append((row['query_id'], dist))
            
        return sorted(distances, key=lambda x: x[1])[:top_n]

    def recommend_for_new_query(self, source_env, target_env, plan_str, benchmark='tpch'):
        """
        Main API for GENERALIZATION:
        Recommend a hint for a query NEVER SEEN BEFORE by the framework.
        """
        # 1. Parse the new plan
        from explain_parser import mysql_explain_parser
        parser = mysql_explain_parser()
        plan_df = parser.parse_explain_analyze(plan_str, 'new_q', 'baseline', 'none', target_env, benchmark)

        # 2. Find similar historical queries to pick candidate hints
        similar = self.find_similar_queries(target_env, plan_df)
        if not similar:
            return "🛡️ KEEP NATIVE: No similar queries found."

        # 3. Collect candidates from the most similar query
        top_sim_qid = similar[0][0]
        logger.info(f"Target query similarity: Found {top_sim_qid} is most similar.")

        # 4. Get hint sets that were successful for similar query in source env
        src_baseline = self._query_baselines.get((source_env, top_sim_qid))
        if src_baseline is None:
            return "🛡️ KEEP NATIVE: No baseline data for similar query in source env."

        candidates = self.df_queries[
            (self.df_queries['env_id'] == source_env) &
            (self.df_queries['query_id'] == top_sim_qid) &
            (self.df_queries['hint_set'] != 'baseline')
        ].copy()
        if candidates.empty:
            return "🛡️ KEEP NATIVE: No hint data for similar query in source env."

        # Compute speedup and sort
        candidates['speedup'] = src_baseline / candidates['execution_time_s'].clip(lower=0.001)
        candidates = candidates.sort_values('speedup', ascending=False).head(3)

        # 5. Filter these candidates through HALO-R logic
        for _, cand in candidates.iterrows():
            hint = cand['hint_set']
            res = self.recommend(top_sim_qid, source_env, target_env, policy='robust')
            if res.recommended_hint == hint:
                return (f"✅ RECOMMEND: {hint} "
                        f"(Based on similarity to {top_sim_qid} and verified SAFE by σ model)")

        return "🛡️ KEEP NATIVE: Candidates from similar queries were rejected by σ model."
       
    def recommend_all(self, source_env: str, target_env: str,
                      policy: str = 'robust') -> List[HaloRecommendation]:
        """Recommend hints for ALL queries in a transfer scenario."""
        query_ids = set()
        for (env, qid, hs) in self._operator_index:
            if env == source_env and hs == 'baseline':
                query_ids.add(qid)

        return [self.recommend(qid, source_env, target_env, policy)
                for qid in sorted(query_ids)]

    def evaluate_transfer(self, source_env: str, target_env: str,
                          policy: str = 'robust') -> dict:
        """
        Evaluate a source → target transfer with the given policy.
        Returns actual speedups and regressions vs target ground truth.
        """
        recs = self.recommend_all(source_env, target_env, policy)

        n_total = n_hinted = n_native = n_regression = 0
        speedups = []
        details = []

        for rec in recs:
            tgt_bl = self._query_baselines.get((target_env, rec.query_id))
            if tgt_bl is None:
                continue
            n_total += 1

            if rec.recommended_hint:
                n_hinted += 1
                ht = self.df_queries[
                    (self.df_queries['env_id'] == target_env) &
                    (self.df_queries['query_id'] == rec.query_id) &
                    (self.df_queries['hint_set'] == rec.recommended_hint)
                ]['execution_time_s']
                if len(ht) > 0:
                    actual_spd = tgt_bl / max(ht.min(), 0.001)
                    if actual_spd < 0.95:
                        n_regression += 1
                else:
                    actual_spd = 1.0
            else:
                n_native += 1
                actual_spd = 1.0

            speedups.append(actual_spd)
            details.append({
                'query_id': rec.query_id,
                'hint': rec.recommended_hint or 'NATIVE',
                'predicted': rec.predicted_speedup,
                'actual': actual_spd,
                'risk': rec.risk_level,
                'fallback': rec.native_fallback,
            })

        return {
            'source': source_env, 'target': target_env,
            'policy': policy,
            'n_queries': n_total, 'n_hinted': n_hinted,
            'n_native': n_native, 'n_regressions': n_regression,
            'avg_speedup': float(np.mean(speedups)) if speedups else 1.0,
            'median_speedup': float(np.median(speedups)) if speedups else 1.0,
            'details': details,
        }

    # ───────────────────────────────────────────────────────────────────
    #  Policy Comparison
    # ───────────────────────────────────────────────────────────────────

    def compare_policies(self, source_env: str, target_env: str) -> dict:
        """
        Compare HALO-G and HALO-R head-to-head on the same transfer.
        Highlights cases where HALO-G regresses but HALO-R saves.
        """
        g = self.evaluate_transfer(source_env, target_env, 'greedy')
        r = self.evaluate_transfer(source_env, target_env, 'robust')

        # Find rescue cases: HALO-G regressed, HALO-R did not
        g_details = {d['query_id']: d for d in g['details']}
        r_details = {d['query_id']: d for d in r['details']}

        rescues = []
        for qid, gd in g_details.items():
            rd = r_details.get(qid)
            if rd and gd['actual'] < 0.95 and rd['actual'] >= 0.95:
                rescues.append({
                    'query_id': qid,
                    'halo_g_hint': gd['hint'],
                    'halo_g_actual': gd['actual'],
                    'halo_r_hint': rd['hint'],
                    'halo_r_actual': rd['actual'],
                })

        return {
            'transfer': f"{source_env} → {target_env}",
            'halo_g': {k: v for k, v in g.items() if k != 'details'},
            'halo_r': {k: v for k, v in r.items() if k != 'details'},
            'rescues': rescues,
            'rescue_count': len(rescues),
        }

    # ───────────────────────────────────────────────────────────────────
    #  Pretty Printing
    # ───────────────────────────────────────────────────────────────────

    def print_recommendation(self, rec: HaloRecommendation):
        policy_label = "HALO-G (Greedy)" if rec.policy == 'greedy' else "HALO-R (Robust)"
        print(f"\n{'='*70}")
        print(f"  [{policy_label}]  {rec.query_id}  |  {rec.source_env} → {rec.target_env}")
        print(f"{'='*70}")
        if rec.recommended_hint:
            print(f"  ✅ RECOMMEND: {rec.recommended_hint}")
            print(f"     Predicted speedup: {rec.predicted_speedup:.2f}x")
            print(f"     Risk: {rec.risk_level}")
        else:
            print(f"  🛡️  KEEP NATIVE (no hint)")
        print(f"  Reason: {rec.reason}")
        if rec.all_candidates:
            print(f"  Candidates:")
            for c in sorted(rec.all_candidates,
                            key=lambda x: x.predicted_target_speedup, reverse=True):
                m = "→" if c.hint_set == rec.recommended_hint else " "
                flag = " ⚠️" if c.n_high_risk > 0 else ""
                print(f"   {m} {c.hint_set}: src={c.source_speedup:.2f}x, "
                      f"pred={c.predicted_target_speedup:.2f}x, "
                      f"ops={c.n_operators_affected} "
                      f"({c.n_benefit}↑ {c.n_high_risk}⚠){flag}")
        print(f"{'='*70}")

    def print_comparison(self, comp: dict):
        print(f"\n{'━'*70}")
        print(f"  HALO Policy Comparison: {comp['transfer']}")
        print(f"{'━'*70}")
        g = comp['halo_g']
        r = comp['halo_r']
        print(f"  {'Metric':<25} {'HALO-G':>12} {'HALO-R':>12} {'Diff':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Avg Speedup':<25} {g['avg_speedup']:>11.3f}x {r['avg_speedup']:>11.3f}x "
              f"{r['avg_speedup']-g['avg_speedup']:>+10.3f}")
        print(f"  {'Hints Applied':<25} {g['n_hinted']:>12} {r['n_hinted']:>12} "
              f"{r['n_hinted']-g['n_hinted']:>+10}")
        print(f"  {'Native Fallback':<25} {g['n_native']:>12} {r['n_native']:>12} "
              f"{r['n_native']-g['n_native']:>+10}")
        print(f"  {'Regressions':<25} {g['n_regressions']:>12} {r['n_regressions']:>12} "
              f"{r['n_regressions']-g['n_regressions']:>+10}")

        if comp['rescues']:
            print(f"\n  🛡️  Rescue Cases (HALO-G failed, HALO-R saved):")
            for rc in comp['rescues']:
                print(f"    {rc['query_id']}: G={rc['halo_g_hint']}→{rc['halo_g_actual']:.2f}x  "
                      f"R={rc['halo_r_hint']}→{rc['halo_r_actual']:.2f}x")
        else:
            print(f"\n  (No rescue cases — both policies had same regression set)")
        print(f"{'━'*70}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════════════

def demo():
    """Full demonstration of the HALO framework."""
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')

    halo = HaloFramework()
    halo.train(df_ops, df_q)

    # ── 1. Single query: compare policies side by side ──
    print("\n" + "▓"*70)
    print("  DEMO 1: Side-by-Side Policy Comparison (Single Query)")
    print("▓"*70)

    for qid in ['tpch_q9', 'tpch_q5', 'tpch_q4']:
        rec_g = halo.recommend(qid, 'A_NVMe', 'B_SATA', policy='greedy')
        rec_r = halo.recommend(qid, 'A_NVMe', 'B_SATA', policy='robust')
        halo.print_recommendation(rec_g)
        halo.print_recommendation(rec_r)

    # ── 2. Full transfer comparison across all scenarios ──
    print("\n" + "▓"*70)
    print("  DEMO 2: Full Policy Comparison Across Transfers")
    print("▓"*70)

    all_envs = sorted(ENV_PROFILES.keys())
    summary_rows = []

    for src in all_envs:
        for tgt in all_envs:
            comp = halo.compare_policies(src, tgt)
            halo.print_comparison(comp)
            summary_rows.append(comp)

    # ── 3. Grand Summary Table ──
    print("\n" + "▓"*70)
    print("  GRAND SUMMARY")
    print("▓"*70)

    total_g_reg = sum(c['halo_g']['n_regressions'] for c in summary_rows)
    total_r_reg = sum(c['halo_r']['n_regressions'] for c in summary_rows)
    total_rescues = sum(c['rescue_count'] for c in summary_rows)
    avg_g_spd = np.mean([c['halo_g']['avg_speedup'] for c in summary_rows])
    avg_r_spd = np.mean([c['halo_r']['avg_speedup'] for c in summary_rows])

    print(f"\n  {'Metric':<30} {'HALO-G':>12} {'HALO-R':>12}")
    print(f"  {'='*55}")
    print(f"  {'Total Regressions (16 xfers)':<30} {total_g_reg:>12} {total_r_reg:>12}")
    print(f"  {'Avg Speedup (all xfers)':<30} {avg_g_spd:>11.3f}x {avg_r_spd:>11.3f}x")
    print(f"  {'Rescue Cases':<30} {'—':>12} {total_rescues:>12}")
    if total_g_reg > 0:
        reduction = (1 - total_r_reg / total_g_reg) * 100
        print(f"  {'Regression Reduction':<30} {'—':>12} {reduction:>10.0f}%")
    print(f"  {'='*55}")

    # Save
    output = {
        'framework': 'HALO v1.0',
        'policies': ['HALO-G (Greedy)', 'HALO-R (Robust)'],
        'grand_summary': {
            'halo_g_total_regressions': total_g_reg,
            'halo_r_total_regressions': total_r_reg,
            'halo_g_avg_speedup': float(avg_g_spd),
            'halo_r_avg_speedup': float(avg_r_spd),
            'total_rescues': total_rescues,
        },
        'transfers': [
            {
                'transfer': c['transfer'],
                'halo_g': c['halo_g'],
                'halo_r': c['halo_r'],
                'rescues': c['rescues'],
            }
            for c in summary_rows
        ]
    }
    with open('/root/halo/results/halo_framework_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to /root/halo/results/halo_framework_results.json")
    return halo


if __name__ == '__main__':
    demo()
