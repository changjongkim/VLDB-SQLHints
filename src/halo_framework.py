# -*- coding: utf-8 -*-
"""
HALO Framework v4 — Hardware-Aware Learned Optimizer
=====================================================

Statistically rigorous hint recommendation using:
  1. Zero-shot Risk Pattern Dictionary (Physical Profile bucketed)
  2. Bottom-up Tree Feature Propagation (Post-order aggregation)
  3. Hardware-Operator Vector Alignment (Demand-Supply Interaction)
  4. Probabilistic MLP with Conformal Calibration (Provable Safety)

Selection Policies:
  HALO-G (Greedy): Experience-based, maximizes peak speedup.
  HALO-R (Robust): Safety-First using Conformal Upper Bounds.
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

import torch
from hw_registry import HW_REGISTRY, compute_hw_features
from workload_metadata import enrich_operator
from risk_pattern_dictionary import RiskPatternDictionary, inject_pattern_risk_features, classify_hw_transition
from tree_propagation import propagate_tree_features
from sigma_model_v4 import (
    HALOProbabilisticMLP, 
    engineer_features_v4, 
    mc_dropout_predict,
    ConformalCalibrator
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('halo_v4')

@dataclass
class OperatorRisk:
    op_type: str
    table_alias: str
    depth: int
    mu: float                   # Predicted mean delta
    sigma: float                # Predicted total uncertainty
    upper_bound: float          # μ + λ*σ (Conformal upper bound)
    risk_level: str             # 'SAFE', 'MODERATE', 'HIGH'
    is_hr: bool

@dataclass
class HintCandidate:
    hint_set: str
    source_speedup: float
    predicted_target_speedup: float
    n_operators: int
    n_high_risk: int
    total_risk_score: float
    operator_details: List[OperatorRisk] = field(default_factory=list)
    expected_gain: float = 1.0
    risk_probability: float = 0.0

@dataclass
class HaloRecommendation:
    query_id: str
    source_env: str
    target_env: str
    policy: str
    recommended_hint: Optional[str]
    speedup: float
    risk_level: str
    reason: str
    details: List[HintCandidate] = field(default_factory=list)

class HaloFramework:
    def __init__(self, model_dir='/root/halo/results/v4'):
        self.model_dir = model_dir
        self.sigma_model = None
        self.scaler = None
        self.calibrator = None
        self.risk_dict = None
        self.feature_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.df_ops = None
        self.df_queries = None
        self._operator_index = {}
        self._query_baselines = {}
        
        # Load v4 model by default
        self.load_model(model_dir)

    def load_model(self, model_dir):
        """Load HALO v4 probabilistic architecture."""
        try:
            # 1. Load Risk Pattern Dictionary
            dict_path = os.path.join(model_dir, 'risk_pattern_dictionary.json')
            self.risk_dict = RiskPatternDictionary()
            if os.path.exists(dict_path):
                self.risk_dict.load(dict_path)
                logger.info(f"Loaded Risk Pattern Dictionary: {dict_path}")

            # 2. Load Results Metadata (feature list)
            res_path = os.path.join(model_dir, 'sigma_results_v4.json')
            if os.path.exists(res_path):
                with open(res_path) as f:
                    res = json.load(f)
                    self.feature_cols = res['feature_list']
                logger.info(f"Loaded model metadata: {res_path}")

            # 3. Load Checkpoint (Model, Scaler, Calibrator)
            model_path = os.path.join(model_dir, 'sigma_model_v4.pt')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Rebuild Model
                input_dim = checkpoint['architecture']['input_dim']
                self.sigma_model = HALOProbabilisticMLP(input_dim=input_dim).to(self.device)
                self.sigma_model.load_state_dict(checkpoint['model_state_dict'])
                self.sigma_model.eval()
                
                # Extract Objects
                self.scaler = checkpoint['scaler']
                self.calibrator = ConformalCalibrator()
                self.calibrator.lambda_calibrated = checkpoint['calibrator_lambda']
                
                logger.info(f"Loaded v4 Checkpoint from: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load HALO v4 model: {e}")

    def ingest_data(self, ops_path, queries_path):
        """Load and index operator data. Selects the best performing run per hint."""
        self.df_ops = pd.read_parquet(ops_path)
        self.df_queries = pd.read_parquet(queries_path)
        
        # 1. Index Queries: Keep track of the best hint_variant per (env, query, hint_set)
        # to ensure we don't mix operators from different runs.
        best_runs = self.df_queries.sort_values('execution_time_s').drop_duplicates(['env_id', 'query_id', 'hint_set'])
        self.best_variants = set(zip(best_runs['env_id'], best_runs['query_id'], best_runs['hint_set'], best_runs.get('hint_variant', '')))
        
        logger.info(f"Indexing {len(self.df_ops)} operators...")
        self._operator_index = {}
        for _, row in self.df_ops.iterrows():
            # Only include operators from the 'best' variant run to avoid duplicates
            run_key = (row['env_id'], row['query_id'], row['hint_set'], row.get('hint_variant', ''))
            if run_key in self.best_variants:
                key = (row['env_id'], row['query_id'], row['hint_set'])
                if key not in self._operator_index:
                    self._operator_index[key] = []
                self._operator_index[key].append(row.to_dict())
        
        # Build query baselines (global minimum per query/env)
        for _, row in self.df_queries[self.df_queries['hint_set'] == 'baseline'].iterrows():
            key = (row['env_id'], row['query_id'])
            if key not in self._query_baselines or row['execution_time_s'] < self._query_baselines[key]:
                self._query_baselines[key] = row['execution_time_s']
        
        logger.info("Data ingestion complete.")

    def recommend(self, query_id: str, source_env: str, 
                  target_env: str, policy: str = 'robust') -> HaloRecommendation:
        """Entry point for hint recommendation."""
        available_hints = self.df_queries[
            (self.df_queries['query_id'] == query_id) & 
            (self.df_queries['env_id'] == source_env)
        ]['hint_set'].unique()
        available_hints = [h for h in available_hints if h != 'baseline']

        if not available_hints:
            return HaloRecommendation(query_id, source_env, target_env, policy, None, 1.0, 'SAFE', "No hints found for query")

        if policy == 'performance':
            return self._recommend_performance(query_id, source_env, target_env, available_hints)
        elif policy == 'robust':
            return self._recommend_robust(query_id, source_env, target_env, available_hints)
        else:
            return self._recommend_greedy(query_id, source_env, target_env, available_hints)

    def _recommend_greedy(self, query_id, source_env, target_env, available_hints):
        """HALO-G: Pick best speedup in source environment."""
        best_hint = None
        max_speedup = 0
        
        baseline = self._query_baselines.get((source_env, query_id), 1.0)
        for h in available_hints:
            times = self.df_queries[
                (self.df_queries['env_id'] == source_env) & 
                (self.df_queries['query_id'] == query_id) & 
                (self.df_queries['hint_set'] == h)
            ]['execution_time_s']
            if len(times) > 0:
                sp = baseline / times.min()
                if sp > max_speedup:
                    max_speedup = sp
                    best_hint = h
        
        return HaloRecommendation(query_id, source_env, target_env, 'greedy', best_hint, max_speedup, 'UNKNOWN', "Greedy selection (Max Speedup)")

    def _recommend_performance(self, query_id, source_env, target_env, available_hints):
        """HALO-P (Performance): Aggressive selection, trusting EPYC-style gains."""
        candidates = []
        for hs in available_hints:
            # We use a lower internal penalty for performance mode
            c = self._assess_hint_risk(query_id, hs, source_env, target_env, mode='performance')
            if c:
                candidates.append(c)

        if not candidates:
            return HaloRecommendation(query_id, source_env, target_env, 'performance', None, 1.0, 'SAFE', "No viable hint candidates")

        candidates.sort(key=lambda x: x.expected_gain, reverse=True)
        best = candidates[0]
        
        # Performance mode threshold: Gain > 1.01 (1% speedup)
        if best.expected_gain > 1.01:
            level = 'GREEN' if best.n_high_risk == 0 else 'YELLOW'
            if best.n_high_risk > 1: level = 'ORANGE'
            
            return HaloRecommendation(query_id, source_env, target_env, 'performance', 
                                     best.hint_set, best.source_speedup, level, 
                                     f"Performance candidate selected (Gain={best.expected_gain:.2f})", candidates)
        
        return HaloRecommendation(query_id, source_env, target_env, 'performance', None, 1.0, 'SAFE', f"Fallback: Low gain {best.expected_gain:.2f}", candidates)

    def _recommend_robust(self, query_id, source_env, target_env, available_hints):
        """HALO-R (v4): Safety-First using Conformal Prediction."""
        candidates = []
        for hs in available_hints:
            c = self._assess_hint_risk(query_id, hs, source_env, target_env)
            if c:
                candidates.append(c)

        if not candidates:
            return HaloRecommendation(query_id, source_env, target_env, 'robust', None, 1.0, 'SAFE', "No viable hint candidates")

        # Sort candidates by expected_gain (speedup adjusted by risk)
        candidates.sort(key=lambda x: x.expected_gain, reverse=True)
        
        # Tier logic
        best = candidates[0]
        
        # If best candidate is risky, try to find a safer one even if it has lower gain
        if best.n_high_risk > 0 or best.risk_probability > 0.3:
            safer = [c for c in candidates if c.n_high_risk == 0 and c.expected_gain > 1.05]
            if safer:
                best = safer[0]
                return HaloRecommendation(query_id, source_env, target_env, 'robust', 
                                         best.hint_set, best.source_speedup, 'GREEN', 
                                         f"Safe alternative candidate selected (Expected Gain={best.expected_gain:.2f})", candidates)

        if best.expected_gain > 1.05:
            level = 'GREEN' if best.n_high_risk == 0 and best.risk_probability < 0.1 else 'YELLOW'
            return HaloRecommendation(query_id, source_env, target_env, 'robust', 
                                     best.hint_set, best.source_speedup, level, 
                                     f"Robust candidate selected (Safety Factor={best.expected_gain/best.source_speedup:.2f})", candidates)
        
        # Fallback to native
        return HaloRecommendation(query_id, source_env, target_env, 'robust', None, 1.0, 'SAFE', f"Fallback to native: Highest expected gain was {best.expected_gain:.2f}", candidates)

    def _assess_hint_risk(self, query_id: str, hint_set: str,
                          source_env: str, target_env: str, mode='robust') -> Optional[HintCandidate]:
        """v4 Risk Assessment: Uses Conformal Upper Bounds (μ + λ*σ)."""
        pair = self._get_env_pair_metadata(source_env, target_env)
        
        baseline_ops = self._operator_index.get((source_env, query_id, 'baseline'), [])
        hinted_ops = self._operator_index.get((source_env, query_id, hint_set), [])
        if not baseline_ops or not hinted_ops:
            return None

        # Phase 2: Structural Propagation on the operator trees
        propagate_tree_features(baseline_ops)
        propagate_tree_features(hinted_ops)

        baseline_time = self._query_baselines.get((source_env, query_id))
        hinted_times = self.df_queries[
            (self.df_queries['env_id'] == source_env) & 
            (self.df_queries['query_id'] == query_id) & 
            (self.df_queries['hint_set'] == hint_set)
        ]['execution_time_s']
        if baseline_time is None or len(hinted_times) == 0:
            return None

        source_speedup = baseline_time / max(hinted_times.min(), 0.001)
        diff_ops = self._diff_operators(baseline_ops, hinted_ops)

        op_risks = []
        n_hr = 0
        total_risk_val = 0.0

        for itm in diff_ops:
            op = itm['operator']
            # Prediction mu, sigma
            mu, sigma = self._predict_operator_probabilistic(op, pair)
            
            # Conformal safety bound (Guarantee: delta <= upper_bound with 90% confidence)
            # Used for hard safety classification (HIGH_RISK)
            upper_bound = self.calibrator.compute_upper_bound(mu, sigma)
            
            # Adaptive Safety Threshold
            base_thresh = self._get_adaptive_threshold(op['op_type'])
            
            is_hr = upper_bound > base_thresh
            if is_hr:
                n_hr += 1
            
            # Risk measure for expected value: μ + 0.5*σ (70th percentile approx)
            # This is less aggressive than the 90th percentile conformal bound
            # to avoid paralyzing the system on complex plans.
            risk_val = max(0, mu + 0.5 * sigma)
            
            # Weighted by time importance
            weight = math.log(max(op['self_time'], 0.001) + 1.1)
            total_risk_val += risk_val * weight

            op_risks.append(OperatorRisk(
                op_type=op['op_type'],
                table_alias=op.get('table_alias', ''),
                depth=op['depth'],
                mu=mu,
                sigma=sigma,
                upper_bound=upper_bound,
                risk_level='HIGH' if is_hr else ('MODERATE' if upper_bound > 0 else 'SAFE'),
                is_hr=is_hr
            ))

        # Expected Gain = Speedup * SafetyFactor
        avg_upper = np.mean([r.upper_bound for r in op_risks]) if op_risks else 0
        risk_probability = min(1.0, n_hr / max(len(op_risks), 1))
        
        # Penalty is normalized by operator complexity to avoid plan-size bias.
        # Performance mode uses a 50% lower penalty factor to be more aggressive.
        penalty_factor = 0.02 if mode == 'performance' else 0.04
        penalty = penalty_factor * total_risk_val / max(math.log(len(op_risks) + 1), 1)
        expected_gain = source_speedup * math.exp(-penalty)

        return HintCandidate(
            hint_set=hint_set,
            source_speedup=source_speedup,
            predicted_target_speedup=source_speedup / math.exp(avg_upper),
            n_operators=len(diff_ops),
            n_high_risk=n_hr,
            total_risk_score=total_risk_val,
            operator_details=op_risks,
            expected_gain=expected_gain,
            risk_probability=risk_probability
        )

    def _predict_operator_probabilistic(self, op: dict, pair: dict) -> Tuple[float, float]:
        """Inference using v4 MLP + MC-Dropout."""
        if self.sigma_model is None:
            return 0.0, 1.0
            
        feat_dict = engineer_features_v4(op, op, pair, self.risk_dict)
        
        # Ensure only model features are used
        X = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        mu, sigma = mc_dropout_predict(self.sigma_model, X_tensor, n_samples=30)
        return float(mu[0]), float(sigma[0])

    def _diff_operators(self, baseline, hinted):
        """Identify operators that are changed or potentially stressed by the hint."""
        # For simplicity, treat all operators in the hinted plan as 'potential points of failure'
        # but prioritize those that differ or have high estimated cost.
        res = []
        for op in hinted:
            if op['self_time'] > 0.1 or op['actual_rows'] > 1000:
                res.append({'operator': op})
        return res

    def _get_env_pair_metadata(self, env1, env2):
        """Compute HW transition features between two environments."""
        return compute_hw_features(env1, env2)

    def _get_adaptive_threshold(self, op_type):
        """v4 Adaptive thresholds (informed by results_v4.json)."""
        # Lower threshold for CPU-bound (aggressive), higher for I/O-bound (permissive)
        thresholds = {
            'TABLE_SCAN': 1.2,
            'INDEX_SCAN': 0.8,
            'HASH_JOIN': 0.8,
            'AGGREGATE': 0.4,
            'SORT': 0.5,
            'FILTER': 1.0,
            'NL_INNER_JOIN': 0.9
        }
        return thresholds.get(op_type, 0.8)

# Singleton helper for quick recommendation
_halo_instance = None
def get_halo_v4():
    global _halo_instance
    if _halo_instance is None:
        _halo_instance = HaloFramework()
        _halo_instance.ingest_data('/root/halo/data/unified_operators.parquet', 
                                   '/root/halo/data/unified_queries.parquet')
    return _halo_instance
