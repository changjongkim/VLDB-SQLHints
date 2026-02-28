# -*- coding: utf-8 -*-
"""
HALO v4 — Phase 1: Zero-shot Risk Pattern Dictionary
=====================================================

CRITICAL FIX: Eliminates data leakage in worst-case features.

Problem (v3):
  The v3 system enriched operators with worst-case features keyed by
  (env_id, query_id). This means:
    1. At inference, a NEVER-SEEN query has no worst-case entry → feature = 0.
    2. A SEEN query has the actual regression outcome baked in → data leakage.
  Reviewers will immediately flag this as "training on the answer".

Solution (v4):
  We build an OFFLINE statistical dictionary keyed by the PHYSICAL PROFILE
  of the operator — not its identity. The key is:
    [op_type, selectivity_bin, data_size_bin, hw_transition_type]

  For each bucket, we precompute:
    P(risk)     — empirical probability of regression in this bucket
    E[penalty]  — expected regression magnitude (log-ratio) if it regresses
    n_samples   — confidence: how many historical samples back this bucket

  At inference, ANY new operator (even from an unseen query on unseen HW)
  can be mapped to its nearest bucket to receive these features.

  This is a ZERO-SHOT SAFE design: no query ID is ever stored or looked up.

Academic Defense:
  "We construct an Offline Risk Profile Dictionary parameterized by operator
   physical attributes and hardware transition type. This eliminates any
   identity-based data leakage while preserving the model's ability to
   leverage historical failure patterns in a zero-shot manner."
"""

import pandas as pd
import numpy as np
import math
import os
import json
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── Paths ──
WORST_REPORTS = {
    'A_NVMe': '/root/kcj_sqlhint/server_a/SQL_hint/NVMe/Summary_Report/worst_nvme_report.csv',
    'A_SATA': '/root/kcj_sqlhint/server_a/SQL_hint/SATA/Summary_Report/worst_sata_report.csv',
}

# ── Hint path → hint_set mapping (reused from v3) ──
HINT_PATH_MAP = {
    'single_hints': 'hint01',
    'table_specific': 'hint02',
    'Hint_02_table_specific': 'hint02',
    'Hint_02_table_sp': 'hint02',
    'Hint_04_join_order_index': 'hint04',
    'Hint_04_join_order': 'hint04',
    'Hint_04_join_ord': 'hint04',
    'Hint_05_index_strategy': 'hint05',
    'Hint_05_index': 'hint05',
    'combinations': 'combo',
}

# ═══════════════════════════════════════════════════════════════════════
#  Binning Functions (State Space Discretization)
# ═══════════════════════════════════════════════════════════════════════

# Selectivity bins: captures how much data the operator produces
# relative to its input (lower = more selective = potentially dangerous
# because it might trigger full scans when selectivity estimate is wrong).
SELECTIVITY_BINS = [0.001, 0.01, 0.1, 0.5, 1.0]  # 6 bins total
SELECTIVITY_LABELS = ['ultra_selective', 'high_selective', 'medium_selective',
                       'low_selective', 'full_pass']

# Data size bins (MB): captures how much data the operator touches
DATA_SIZE_BINS = [1, 10, 100, 1000, 10000]  # 6 bins total
DATA_SIZE_LABELS = ['tiny', 'small', 'medium', 'large', 'huge']

# HW transition types: groups the continuous HW delta into categories
HW_TRANSITION_TYPES = [
    'storage_downgrade',     # NVMe → SATA (IOPS drops significantly)
    'storage_upgrade',       # SATA → NVMe
    'compute_downgrade',     # High-GHz → Low-GHz
    'compute_upgrade',       # Low-GHz → High-GHz
    'both_downgrade',        # Storage + Compute both drop
    'both_upgrade',          # Storage + Compute both increase
    'mixed_transition',      # One up, one down
    'same_hw',               # No significant change
]


def bin_selectivity(selectivity: float) -> str:
    """Bin a selectivity value into a discrete category."""
    if selectivity <= 0.001:
        return 'ultra_selective'
    elif selectivity <= 0.01:
        return 'high_selective'
    elif selectivity <= 0.1:
        return 'medium_selective'
    elif selectivity <= 0.5:
        return 'low_selective'
    else:
        return 'full_pass'


def bin_data_size(size_mb: float) -> str:
    """Bin a data size (MB) into a discrete category."""
    if size_mb <= 1:
        return 'tiny'
    elif size_mb <= 10:
        return 'small'
    elif size_mb <= 100:
        return 'medium'
    elif size_mb <= 1000:
        return 'large'
    else:
        return 'huge'


def classify_hw_transition(hw_features: dict) -> str:
    """Classify a hardware transition into a type category.
    
    Args:
        hw_features: dict from compute_hw_features() with iops_ratio, cpu_clock_ratio, etc.
    """
    storage_down = hw_features.get('is_storage_downgrade', 0)
    cpu_down = hw_features.get('is_cpu_downgrade', 0)
    
    iops_ratio = hw_features.get('iops_ratio', 0.0)
    clock_ratio = hw_features.get('cpu_clock_ratio', 0.0)
    
    # Significant thresholds (log-ratio)
    SIGNIFICANT = 0.2  # ~22% change
    
    storage_significant = abs(iops_ratio) > SIGNIFICANT
    compute_significant = abs(clock_ratio) > SIGNIFICANT
    
    if not storage_significant and not compute_significant:
        return 'same_hw'
    
    if storage_significant and compute_significant:
        if storage_down and cpu_down:
            return 'both_downgrade'
        elif not storage_down and not cpu_down:
            return 'both_upgrade'
        else:
            return 'mixed_transition'
    
    if storage_significant:
        return 'storage_downgrade' if storage_down else 'storage_upgrade'
    
    return 'compute_downgrade' if cpu_down else 'compute_upgrade'


# ═══════════════════════════════════════════════════════════════════════
#  Risk Pattern Dictionary Builder
# ═══════════════════════════════════════════════════════════════════════

class RiskPatternDictionary:
    """
    Offline-compiled statistical risk dictionary.
    
    The dictionary maps operator physical profiles to historical risk statistics.
    Key: (op_type, selectivity_bin, data_size_bin, hw_transition_type)
    Value: {p_risk, e_penalty, n_samples, max_penalty}
    
    This is the CORE contribution to fixing the data leakage problem.
    No query IDs are stored — only physical operator characteristics.
    """
    
    def __init__(self):
        self.dictionary: Dict[Tuple[str, str, str, str], dict] = {}
        self._is_compiled = False
        # Precomputed global fallback statistics
        self._global_p_risk = 0.0
        self._global_e_penalty = 0.0
    
    def compile_from_training_data(self,
                                    df_operators: pd.DataFrame,
                                    df_delta_labels: pd.DataFrame,
                                    hw_transition_types: Dict[str, str]):
        """
        Build the risk dictionary from training operator pairs.
        
        Args:
            df_operators: operator-level data with op_type, actual_rows, est_rows, 
                          table_size_mb, etc.
            df_delta_labels: DataFrame with columns [query_id, op_type, table_alias,
                             env1, env2, delta_time, ...] — the labeled training pairs
            hw_transition_types: mapping from pair_label → hw_transition_type string
        
        Instead of keying by (env_id, query_id), we key by 
        (op_type, selectivity_bin, data_size_bin, hw_transition_type).
        """
        logger.info("Compiling Risk Pattern Dictionary...")
        
        # Build entries
        bucket_data = {}  # key → list of delta values
        
        for _, row in df_delta_labels.iterrows():
            # Compute selectivity
            actual_rows = max(row.get('log_actual_rows', 0), 0)
            est_rows = max(row.get('log_est_rows', 0), 0)
            # Approximate selectivity from the exponential of the log values
            try:
                selectivity = math.exp(actual_rows) / max(math.exp(est_rows), 1)
            except OverflowError:
                selectivity = 1.0
            selectivity = min(selectivity, 10.0)
            sel_bin = bin_selectivity(selectivity)
            
            # Data size bin
            table_size_mb = math.exp(row.get('log_table_size_mb', 0))
            size_bin = bin_data_size(table_size_mb)
            
            # HW transition type
            pair_label = row.get('pair_label', '')
            hw_type = hw_transition_types.get(pair_label, 'same_hw')
            
            # Operator type
            op_type = row.get('op_type', 'OTHER')
            
            # Bucket key — NO QUERY ID involved
            key = (op_type, sel_bin, size_bin, hw_type)
            
            if key not in bucket_data:
                bucket_data[key] = []
            bucket_data[key].append(row.get('delta_time', 0.0))
        
        # Compute statistics per bucket
        all_deltas = []
        for key, deltas in bucket_data.items():
            deltas = np.array(deltas)
            all_deltas.extend(deltas)
            
            # An operator "regressed" if delta > 0.5 (50%+ slowdown)
            regression_mask = deltas > 0.5
            p_risk = float(regression_mask.mean())
            
            # Expected penalty: average delta among regressed samples
            if regression_mask.any():
                e_penalty = float(deltas[regression_mask].mean())
            else:
                e_penalty = 0.0
            
            self.dictionary[key] = {
                'p_risk': round(p_risk, 4),
                'e_penalty': round(e_penalty, 4),
                'n_samples': len(deltas),
                'max_penalty': round(float(deltas.max()), 4),
                'mean_delta': round(float(deltas.mean()), 4),
                'std_delta': round(float(deltas.std()), 4),
            }
        
        # Global fallback
        all_deltas = np.array(all_deltas)
        self._global_p_risk = float((all_deltas > 0.5).mean())
        self._global_e_penalty = float(all_deltas[all_deltas > 0.5].mean()) if (all_deltas > 0.5).any() else 0.0
        
        self._is_compiled = True
        logger.info(f"  Risk Dictionary compiled: {len(self.dictionary)} buckets "
                     f"from {len(all_deltas)} training pairs")
        logger.info(f"  Global baseline: P(risk)={self._global_p_risk:.3f}, "
                     f"E[penalty]={self._global_e_penalty:.3f}")
    
    def lookup(self,
               op_type: str,
               selectivity: float,
               table_size_mb: float,
               hw_transition_type: str) -> dict:
        """
        Look up risk statistics for an operator profile.
        
        This is the INFERENCE-TIME function. It takes ONLY physical
        operator characteristics — never a query ID.
        
        Returns:
            dict with keys: pattern_risk_prob, pattern_expected_penalty,
                            pattern_confidence (n_samples backing this bucket)
        """
        sel_bin = bin_selectivity(selectivity)
        size_bin = bin_data_size(table_size_mb)
        key = (op_type, sel_bin, size_bin, hw_transition_type)
        
        # Exact match
        if key in self.dictionary:
            entry = self.dictionary[key]
            return {
                'pattern_risk_prob': entry['p_risk'],
                'pattern_expected_penalty': entry['e_penalty'],
                'pattern_confidence': min(entry['n_samples'] / 50.0, 1.0),  # Normalize
            }
        
        # Fallback 1: Relax data_size_bin (match on op_type + selectivity + hw)
        for fallback_size in DATA_SIZE_LABELS:
            fallback_key = (op_type, sel_bin, fallback_size, hw_transition_type)
            if fallback_key in self.dictionary:
                entry = self.dictionary[fallback_key]
                return {
                    'pattern_risk_prob': entry['p_risk'] * 0.8,  # Discount for inexact match
                    'pattern_expected_penalty': entry['e_penalty'] * 0.8,
                    'pattern_confidence': min(entry['n_samples'] / 50.0, 1.0) * 0.5,
                }
        
        # Fallback 2: Match on op_type + hw_transition_type only
        matches = [v for k, v in self.dictionary.items()
                   if k[0] == op_type and k[3] == hw_transition_type]
        if matches:
            avg_risk = np.mean([m['p_risk'] for m in matches])
            avg_penalty = np.mean([m['e_penalty'] for m in matches])
            total_n = sum(m['n_samples'] for m in matches)
            return {
                'pattern_risk_prob': float(avg_risk * 0.6),
                'pattern_expected_penalty': float(avg_penalty * 0.6),
                'pattern_confidence': min(total_n / 100.0, 1.0) * 0.3,
            }
        
        # Fallback 3: Global prior (very low confidence)
        return {
            'pattern_risk_prob': self._global_p_risk * 0.3,
            'pattern_expected_penalty': self._global_e_penalty * 0.3,
            'pattern_confidence': 0.1,
        }
    
    def save(self, path: str):
        """Save the compiled dictionary to disk."""
        data = {
            'dictionary': {str(k): v for k, v in self.dictionary.items()},
            'global_p_risk': self._global_p_risk,
            'global_e_penalty': self._global_e_penalty,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Risk Pattern Dictionary saved to {path}")
    
    def load(self, path: str):
        """Load a pre-compiled dictionary."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.dictionary = {eval(k): v for k, v in data['dictionary'].items()}
        self._global_p_risk = data['global_p_risk']
        self._global_e_penalty = data['global_e_penalty']
        self._is_compiled = True
        logger.info(f"Risk Pattern Dictionary loaded: {len(self.dictionary)} buckets")


# ═══════════════════════════════════════════════════════════════════════
#  Feature Injection (replaces old enrich_operators_with_worst_case)
# ═══════════════════════════════════════════════════════════════════════

def inject_pattern_risk_features(features: dict,
                                  op_type: str,
                                  selectivity: float,
                                  table_size_mb: float,
                                  hw_transition_type: str,
                                  risk_dict: RiskPatternDictionary) -> dict:
    """
    Inject zero-shot pattern risk features into an operator's feature dict.
    
    This REPLACES the old v3 worst-case features:
      OLD (LEAKING):  has_worst_case, worst_regression_ratio, log_worst_regression,
                       worst_bp_reads_ratio, worst_hint_matches
      NEW (ZERO-SHOT): pattern_risk_prob, pattern_expected_penalty, pattern_confidence
    
    The new features depend ONLY on the operator's physical profile,
    NOT on whether this specific query was ever seen before.
    """
    risk = risk_dict.lookup(op_type, selectivity, table_size_mb, hw_transition_type)
    features['pattern_risk_prob'] = risk['pattern_risk_prob']
    features['pattern_expected_penalty'] = risk['pattern_expected_penalty']
    features['pattern_confidence'] = risk['pattern_confidence']
    return features


# ═══════════════════════════════════════════════════════════════════════
#  Demo / Standalone
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("HALO v4 — Risk Pattern Dictionary (Zero-shot)")
    print("=" * 60)
    
    rpd = RiskPatternDictionary()
    
    # Example: simulate some training data
    print("\nExample bucket lookups (before compilation — returns global prior):")
    result = rpd.lookup('TABLE_SCAN', 0.005, 22000, 'storage_downgrade')
    print(f"  TABLE_SCAN, high_selective, huge, storage_downgrade → {result}")
    
    result = rpd.lookup('HASH_JOIN', 0.5, 50, 'compute_downgrade')
    print(f"  HASH_JOIN, low_selective, medium, compute_downgrade → {result}")
    
    print("\n✅ Risk Pattern Dictionary module ready.")
    print("   Run sigma_model_v4.py to compile the dictionary from training data.")
