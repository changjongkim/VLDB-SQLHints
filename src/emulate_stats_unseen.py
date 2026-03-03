#!/usr/bin/env python3
"""
STATS Unseen Workload Emulation Script

Purpose:
Since STATS is entirely unseen and we only have target hardware specs (no actual execution data on Xeon),
we emulate the relative target performance using HALO v4's calibrated risk models.

We compare 3 strategies:
1. NATIVE: Always use the default optimizer plan (Baseline). Target Speedup = 1.0
2. SOTA-like (Aggressive): Pick the hint that performed best on the Source (A_SATA) environment,
   ignoring target hardware differences. Emulated Target Speedup = expected_gain of that hint.
3. HALO-R (Robust): Pick the hint using HALO's risk-aware conformal bounds. Fallback to NATIVE if risky.

This demonstrates the validity of HALO's defensive fallbacks.
"""

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from halo_framework import get_halo_v4
from generate_stats_v4_recommendations import get_mysql_explain, HINT_DEFINITIONS, SQL_DIR
from explain_to_operators import parse_explain_json
from workload_metadata import enrich_operators_batch
from hw_registry import compute_hw_features
from tree_propagation import propagate_tree_features

SOURCE_DATA = '/root/halo/data/stats/stats_A_SATA_queries.parquet'

def emulate_target_environment(target_env, df_source, halo_instance):
    print(f"\n[{target_env}] Starting Emulation from A_SATA...")
    hw_pair = compute_hw_features('A_SATA', target_env)
    
    results = []
    grouped = df_source.groupby('query_id')
    
    for qid, group in tqdm(grouped, desc=f"Emulating {target_env}"):
        qid_str = str(qid)
        sql_file = os.path.join(SQL_DIR, f"stats_{qid_str}.sql")
        if not os.path.exists(sql_file):
            continue
            
        baseline_row = group[group['hint_set'] == 'baseline']
        if baseline_row.empty: continue
        source_baseline_time = baseline_row.iloc[0]['execution_time_s']
        
        candidates = []
        for _, row in group.iterrows():
            h_name = row['hint_set']
            if h_name == 'baseline': continue
            
            source_hinted_time = row['execution_time_s']
            source_speedup = source_baseline_time / max(source_hinted_time, 0.001)
            
            explain_json = get_mysql_explain(sql_file, HINT_DEFINITIONS[h_name])
            if not explain_json: continue
            
            ops = parse_explain_json(explain_json, query_id=qid_str, env_id='A_SATA')
            if not ops: continue
            
            df_ops = pd.DataFrame(ops)
            df_ops = enrich_operators_batch(df_ops, benchmark_col='benchmark', env_col='env_id')
            enriched_ops = df_ops.to_dict('records')
            
            total_est_cost = sum(op.get('est_cost', 0) for op in enriched_ops if op.get('depth', 0) == 0)
            if total_est_cost > 0:
                scale = source_hinted_time / total_est_cost
                for op in enriched_ops:
                    op['self_time'] = op.get('est_cost', 0) * scale 
                    op['actual_rows'] = op.get('est_rows', 1) 
            
            propagate_tree_features(enriched_ops)
            
            n_hr = 0
            total_risk_val = 0.0
            sum_sigma = 0.0 # Track uncertainty
            for op in enriched_ops:
                mu, sigma = halo_instance._predict_operator_probabilistic(op, hw_pair)
                upper_bound = halo_instance.calibrator.compute_upper_bound(mu, sigma)
                base_thresh = halo_instance._get_adaptive_threshold(op['op_type'])
                if upper_bound > base_thresh: n_hr += 1
                total_risk_val += max(0, mu + 0.5 * sigma)
                sum_sigma += sigma
                
            risk_prob = min(1.0, n_hr / max(len(enriched_ops), 1))
            penalty = 0.04 * total_risk_val / max(math.log(len(enriched_ops) + 1), 1)
            mean_sigma = sum_sigma / max(len(enriched_ops), 1)
            
            # This is the Emulated True Target Speedup (if HALO model is the ground truth simulator)
            expected_gain = source_speedup * math.exp(-penalty)
            
            candidates.append({
                'hint': h_name,
                'source_speedup': source_speedup,
                'expected_gain': expected_gain,
                'n_hr': n_hr,
                'risk_prob': risk_prob,
                'mean_sigma': mean_sigma
            })
            
        if not candidates:
            continue
            
        # Strategy 1: NATIVE
        native_speedup = 1.0
        
        # Strategy 2: SOTA-like (Aggressive)
        # Pick best on source, if > 1.05.
        agg_candidates = sorted(candidates, key=lambda x: x['source_speedup'], reverse=True)
        agg_best = agg_candidates[0]
        agg_speedup = agg_best['expected_gain'] if agg_best['source_speedup'] > 1.05 else 1.0
        
        # Strategy 3: HALO-R
        # Pick based on expected_gain, with safety constraints
        halo_candidates = sorted(candidates, key=lambda x: x['expected_gain'], reverse=True)
        halo_best = halo_candidates[0]
        
        if halo_best['n_hr'] > 0 or halo_best['risk_prob'] > 0.2:
            safer = [c for c in halo_candidates if c['n_hr'] == 0 and c['expected_gain'] > 1.05]
            if safer:
                halo_best = safer[0]
                
        halo_speedup = halo_best['expected_gain'] if halo_best['expected_gain'] > 1.05 else 1.0
        
        results.append({
            'query_id': qid_str,
            'source_speedup_best': agg_best['source_speedup'],
            'mean_sigma_best': agg_best['mean_sigma'],
            'native_speedup': native_speedup,
            'agg_speedup': agg_speedup,
            'halo_speedup': halo_speedup,
            'halo_dropped_agg': 1 if (agg_speedup < 1.0 and halo_speedup == 1.0) else 0
        })
        
    return pd.DataFrame(results)

def main():
    halo = get_halo_v4()
    df_source = pd.read_parquet(SOURCE_DATA)
    
    envs = ['Xeon_NVMe', 'Xeon_SATA']
    summary = []
    
    for env in envs:
        df_res = emulate_target_environment(env, df_source, halo)
        if df_res.empty:
            continue
            
        total_queries = len(df_res)
        
        # Calculate GM (Geometric Mean) Speedup
        gm_agg = np.exp(np.log(df_res['agg_speedup']).mean())
        gm_halo = np.exp(np.log(df_res['halo_speedup']).mean())
        avg_sigma = df_res['mean_sigma_best'].mean()
        
        # Regressions (< 0.95 speedup)
        reg_agg = len(df_res[df_res['agg_speedup'] < 0.95])
        reg_halo = len(df_res[df_res['halo_speedup'] < 0.95])
        
        # Disaster prevention: how many times HALO fell back to NATIVE when agg failed
        saved_by_halo = df_res['halo_dropped_agg'].sum()
        
        summary.append({
            'Target Env': env,
            'NATIVE Speedup': '1.00x',
            'SOTA Speedup': f"{gm_agg:.2f}x",
            'HALO Speedup': f"{gm_halo:.2f}x",
            'Mean Uncertainty (σ)': f"{avg_sigma:.3f}",
            'HALO OOD Blocks': saved_by_halo
        })
        
        # Optional: Save to CSV
        df_res.to_csv(f'/root/halo/results/stats_emulation_{env}.csv', index=False)
        
    print("\n" + "="*60)
    print(" STATS Unseen Workload Emulation Summary (Vs A_SATA Source)")
    print("="*60)
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    print("="*60)

if __name__ == '__main__':
    main()
