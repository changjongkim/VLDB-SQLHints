# -*- coding: utf-8 -*-
"""
HALO Worst-Case Data Integration
==================================

Integrates worst-case experiment data (queries where hints caused regressions)
into the HALO framework. This data provides critical "negative examples" that
help the σ model learn WHAT MAKES A HINT DANGEROUS.

Data sources:
  - worst_nvme_report.csv: NVMe env worst-performing hint per query
  - worst_sata_report.csv: SATA env worst-performing hint per query

Each row contains:
  - Base_WallTime / Hint_WallTime (regression ratio)
  - Base_BP_Reads / Hint_BP_Reads (buffer pool disk reads)
  - Disk utilization and wait times
  - Applied_Hint_Path (which hint was the worst)
"""

import pandas as pd
import numpy as np
import math
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── Paths ──
WORST_REPORTS = {
    'A_NVMe': '/root/kcj_sqlhint/server_a/SQL_hint/NVMe/Summary_Report/worst_nvme_report.csv',
    'A_SATA': '/root/kcj_sqlhint/server_a/SQL_hint/SATA/Summary_Report/worst_sata_report.csv',
}

# ── Hint path → hint_set mapping ──
# Map the Applied_Hint_Path from worst-case reports to our standardized hint_set IDs
HINT_PATH_MAP = {
    'single_hints': 'hint01',        # single optimizer switch
    'table_specific': 'hint02',      # table-specific hints (BKA, MRR)
    'Hint_02_table_specific': 'hint02',
    'Hint_02_table_sp': 'hint02',
    'Hint_04_join_order_index': 'hint04',
    'Hint_04_join_order': 'hint04',
    'Hint_04_join_ord': 'hint04',
    'Hint_05_index_strategy': 'hint05',
    'Hint_05_index': 'hint05',
    'combinations': 'combo',          # combination hints
}


def _map_hint_path_to_set(hint_path: str) -> str:
    """Map a worst-case hint file path to a standardized hint_set ID."""
    if pd.isna(hint_path):
        return 'unknown'
    for prefix, hint_set in HINT_PATH_MAP.items():
        if prefix in hint_path:
            return hint_set
    return 'unknown'


def _normalize_query_id(category: str, query: str) -> str:
    """Normalize query IDs to match unified_queries format."""
    query = str(query).strip()
    if category == 'TPCH':
        return f'tpch_q{query}'
    else:
        return query  # JOB queries are already like '10a', '11d', etc.


def load_worst_case_data() -> pd.DataFrame:
    """
    Load and normalize all worst-case experiment data.
    
    Returns DataFrame with columns:
      env_id, benchmark, query_id, hint_set, hint_path,
      base_walltime, hint_walltime, regression_ratio,
      base_bp_reads, hint_bp_reads, bp_reads_ratio,
      base_disk_util, hint_disk_util, base_disk_wait, hint_disk_wait
    """
    all_rows = []
    
    for env_id, csv_path in WORST_REPORTS.items():
        if not os.path.exists(csv_path):
            logger.warning(f"Worst-case report not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        for _, row in df.iterrows():
            query_id = _normalize_query_id(row['Category'], row['Query'])
            hint_set = _map_hint_path_to_set(row.get('Applied_Hint_Path'))
            
            base_wt = row.get('Base_WallTime')
            hint_wt = row.get('Hint_WallTime')
            
            # Compute regression ratio (>1 = hint is slower)
            if pd.notna(base_wt) and pd.notna(hint_wt) and base_wt > 0:
                regression_ratio = hint_wt / base_wt
            else:
                regression_ratio = np.nan
            
            # Buffer pool reads ratio
            base_bp = row.get('Base_BP_Reads')
            hint_bp = row.get('Hint_BP_Reads')
            if pd.notna(base_bp) and pd.notna(hint_bp) and base_bp > 0:
                bp_ratio = hint_bp / base_bp
            else:
                bp_ratio = np.nan
            
            all_rows.append({
                'env_id': env_id,
                'benchmark': row['Category'],
                'query_id': query_id,
                'hint_set': hint_set,
                'hint_path': row.get('Applied_Hint_Path', ''),
                'base_walltime': base_wt,
                'hint_walltime': hint_wt,
                'regression_ratio': regression_ratio,
                'base_bp_reads': base_bp,
                'hint_bp_reads': hint_bp,
                'bp_reads_ratio': bp_ratio,
                'base_disk_util': row.get('Base_Avg_Disk_UtilP'),
                'hint_disk_util': row.get('Hint_Avg_Disk_UtilP'),
                'base_disk_wait': row.get('Base_Avg_Disk_Wait'),
                'hint_disk_wait': row.get('Hint_Avg_Disk_Wait'),
                'is_worst_case': 1,  # Flag for the σ model
            })
    
    df_worst = pd.DataFrame(all_rows)
    logger.info(f"Total worst-case entries: {len(df_worst)}")
    return df_worst


def build_worst_case_query_features(df_worst: pd.DataFrame) -> dict:
    """
    Build a lookup dict: (env_id, query_id) → worst-case features.
    
    These features can be merged into the σ model training data
    to tell the model: "this query has known regression patterns."
    """
    features = {}
    for _, row in df_worst.iterrows():
        key = (row['env_id'], row['query_id'])
        features[key] = {
            'worst_regression_ratio': row['regression_ratio'],
            'worst_hint': row['hint_set'],
            'worst_bp_reads_ratio': row['bp_reads_ratio'],
            'worst_disk_util': row.get('hint_disk_util', np.nan),
            'worst_disk_wait': row.get('hint_disk_wait', np.nan),
            'has_worst_case': 1,
            'log_worst_regression': math.log(max(row['regression_ratio'], 0.01))
                                    if pd.notna(row['regression_ratio']) else 0.0,
        }
    return features


def enrich_operators_with_worst_case(df_ops: pd.DataFrame, 
                                      df_worst: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich operators DataFrame with worst-case query-level features.
    
    For each operator, if its (env_id, query_id) has known worst-case data,
    we add features that tell the σ model:
      - This query has a known regression pattern
      - The regression severity (ratio)
      - The BP read overhead
    
    For operators with NO worst-case data, features default to 0 (no known risk).
    """
    df_ops = df_ops.copy()
    
    # Build lookup
    wc_features = build_worst_case_query_features(df_worst)
    
    # Initialize new columns
    new_cols = ['has_worst_case', 'worst_regression_ratio', 
                'log_worst_regression', 'worst_bp_reads_ratio',
                'worst_hint_matches']
    for col in new_cols:
        df_ops[col] = 0.0
    
    enriched = 0
    for idx, row in df_ops.iterrows():
        key = (row['env_id'], row['query_id'])
        if key in wc_features:
            wc = wc_features[key]
            df_ops.at[idx, 'has_worst_case'] = 1
            df_ops.at[idx, 'worst_regression_ratio'] = wc.get('worst_regression_ratio', 0)
            df_ops.at[idx, 'log_worst_regression'] = wc.get('log_worst_regression', 0)
            df_ops.at[idx, 'worst_bp_reads_ratio'] = wc.get('worst_bp_reads_ratio', 0)
            # Is this operator's hint_set the same as the worst one?
            df_ops.at[idx, 'worst_hint_matches'] = 1.0 if row.get('hint_set') == wc.get('worst_hint') else 0.0
            enriched += 1
    
    logger.info(f"Enriched {enriched:,} operators with worst-case features "
                f"(out of {len(df_ops):,} total)")
    return df_ops


# ═══════════════════════════════════════════════════════════════════════
#  Quick Demo / Standalone Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df_worst = load_worst_case_data()
    
    print("\n" + "="*70)
    print("HALO Worst-Case Data Summary")
    print("="*70)
    
    for env in df_worst['env_id'].unique():
        env_data = df_worst[df_worst['env_id'] == env]
        has_regression = env_data[env_data['regression_ratio'] > 1.05]
        
        print(f"\n[{env}] Total: {len(env_data)} queries")
        print(f"  With regression (>5%): {len(has_regression)}")
        
        if len(has_regression) > 0:
            print(f"  Worst regression: {has_regression['regression_ratio'].max():.2f}x "
                  f"({has_regression.loc[has_regression['regression_ratio'].idxmax(), 'query_id']})")
            print(f"  Avg regression: {has_regression['regression_ratio'].mean():.2f}x")
        
        for bm in env_data['benchmark'].unique():
            bm_data = env_data[env_data['benchmark'] == bm]
            print(f"  [{bm}] {len(bm_data)} queries, "
                  f"hint distribution: {bm_data['hint_set'].value_counts().to_dict()}")
    
    print("\n✅ Worst-case data loaded and ready for integration.")
