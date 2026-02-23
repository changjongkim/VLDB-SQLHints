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


# ═══════════════════════════════════════════════════════════════════════
#  Hardware Environment Registry
# ═══════════════════════════════════════════════════════════════════════

ENV_PROFILES = {
    'A_NVMe': {'server': 'A', 'storage': 'NVMe', 'cpu': 'i9-13900K'},
    'A_SATA': {'server': 'A', 'storage': 'SATA', 'cpu': 'i9-13900K'},
    'B_NVMe': {'server': 'B', 'storage': 'NVMe', 'cpu': 'EPYC-7453'},
    'B_SATA': {'server': 'B', 'storage': 'SATA', 'cpu': 'EPYC-7453'},
}


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
    RISK_HIGH_THRESHOLD = 0.5
    RISK_MODERATE_THRESHOLD = 0.2
    MIN_SPEEDUP_THRESHOLD = 1.02
    MAX_HIGH_RISK_OPS = 0        # zero tolerance

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
              model_path: str = '/root/halo/results/sigma_model_v2.pkl'):
        """
        Load profiling data and the pre-trained σ model.

        Args:
            df_operators: unified_operators.parquet
            df_queries:   unified_queries.parquet
            model_path:   path to σ model pickle (from sigma_model_v2.py)
        """
        logger.info("═══ HALO Framework: Initializing ═══")

        self.df_operators = df_operators.copy()
        self.df_queries = df_queries.copy()

        # Load σ model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            self.sigma_model = saved['model']
            self.feature_cols = saved['feature_cols']
            logger.info(f"  σ model loaded: {len(self.feature_cols)} features")
        else:
            logger.warning(f"  σ model not found at {model_path}. "
                           "HALO-R will be unavailable. Run sigma_model_v2.py first.")

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
        src = ENV_PROFILES.get(source_env, {})
        tgt = ENV_PROFILES.get(target_env, {})
        return {
            'storage_changed': int(src.get('storage') != tgt.get('storage')),
            'compute_changed': int(src.get('cpu') != tgt.get('cpu')),
        }

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

        features['depth'] = op.get('depth', 0)
        features['num_children'] = op.get('num_children', 0)

        features['storage_changed'] = hw_change['storage_changed']
        features['compute_changed'] = hw_change['compute_changed']
        features['both_changed'] = hw_change['storage_changed'] * hw_change['compute_changed']
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

    def _assess_hint_risk(self, query_id: str, hint_set: str,
                          source_env: str, target_env: str) -> Optional[HintCandidate]:
        """Core OLHS logic: assess operator-level risk of a hint on target HW."""
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
            delta = self._predict_operator_delta(op, hw_change)
            abs_delta = abs(delta)
            if abs_delta > self.RISK_HIGH_THRESHOLD:
                risk_level = 'HIGH'; n_high += 1
            elif abs_delta > self.RISK_MODERATE_THRESHOLD:
                risk_level = 'MODERATE'
            else:
                risk_level = 'SAFE'
            contribution = 'HURT' if delta > 0 else 'BENEFIT'
            if delta <= 0: n_benefit += 1
            weight = math.log(max(op.get('self_time', 0.001), 0.001) + 1) + 1
            total_risk_score += abs_delta * weight
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

        return HintCandidate(
            hint_set=hint_set,
            source_speedup=source_speedup,
            predicted_target_speedup=predicted_target_speedup,
            n_operators_affected=len(diffs),
            n_high_risk=n_high,
            n_benefit=n_benefit,
            total_risk_score=total_risk_score,
            operator_details=op_risks,
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
        HALO-R: Evaluate every hint using σ model, reject high-risk ones.
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

        # Filter: reject high-risk candidates
        safe = [c for c in candidates
                if c.n_high_risk <= self.MAX_HIGH_RISK_OPS
                and c.predicted_target_speedup >= self.MIN_SPEEDUP_THRESHOLD]

        if safe:
            best = max(safe, key=lambda c: c.predicted_target_speedup)
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
                reason=(f"HALO-R selected {best.hint_set}: "
                        f"source {best.source_speedup:.2f}x → "
                        f"predicted target {best.predicted_target_speedup:.2f}x. "
                        f"{best.n_operators_affected} ops affected "
                        f"({best.n_benefit}↑, {best.n_high_risk}⚠). "
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
                        f"Best was {best_rej.hint_set} (src {best_rej.source_speedup:.2f}x) "
                        f"but {best_rej.n_high_risk} high-risk ops detected. "
                        f"Keeping native to avoid regression."),
                all_candidates=candidates,
                native_fallback=True,
            )

    # ───────────────────────────────────────────────────────────────────
    #  Batch Operations
    # ───────────────────────────────────────────────────────────────────

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
