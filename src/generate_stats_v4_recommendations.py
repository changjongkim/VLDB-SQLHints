#!/usr/bin/env python3
"""
HALO v4 Recommendation Engine for STATS Benchmark (Xeon NVMe/SATA)
Source Environment: A_SATA (Execution data collected locally)
Target Environments: Xeon_NVMe, Xeon_SATA
"""

import os
import json
import pandas as pd
import numpy as np
import math
import glob
from tqdm import tqdm

# HALO internal modules
from halo_framework import get_halo_v4, HintCandidate, OperatorRisk
from explain_to_operators import parse_explain_json
from workload_metadata import enrich_operators_batch
from hw_registry import compute_hw_features
from tree_propagation import propagate_tree_features

# Paths
SOURCE_DATA = '/root/halo/data/stats/stats_A_SATA_queries.parquet'
SQL_DIR = '/root/stats-benchmark/queries_mysql'
OUTPUT_BASE = '/root/stats-benchmark/queries_mysql_halop'

HINT_DEFINITIONS = {
    'baseline': '',
    'hint01': '/*+ SET_VAR(optimizer_switch="block_nested_loop=off") */',
    'hint02': '/*+ BKA(v p c ph pl u b t) */',
    'hint04': '/*+ JOIN_ORDER(v, p, c, ph, pl, u, b, t) */',
    'hint05': '/*+ SET_VAR(optimizer_switch="block_nested_loop=off") */',
}

def get_mysql_explain(sql_path, hint_comment):
    # We use a mocked EXPLAIN call for speed here since we already have structures
    # but in a real scenario we'd call mysql. For STATS, let's reuse our parser
    # directly on the baseline/hinted EXPLAIN JSON.
    # To keep it robust, we'll call the actual mysql once per query/hint to get structure.
    from generate_stats_hinted_queries import get_explain_plan
    with open(sql_path, 'r') as f:
        sql = f.read()
    return get_explain_plan(sql, hint_comment)

def recommend_for_scenario(target_env, df_source, halo_instance):
    print(f"\n>>> Generating recommendations for Target: {target_env} (Source: A_SATA) ...")
    
    hw_pair = compute_hw_features('A_SATA', target_env)
    results = []
    
    # Group source data by query
    grouped = df_source.groupby('query_id')
    
    for qid, group in tqdm(grouped, desc=f"Processing {target_env}"):
        qid_str = str(qid) # e.g. "q1"
        sql_file = os.path.join(SQL_DIR, f"stats_{qid_str}.sql")
        if not os.path.exists(sql_file):
            continue
            
        # 1. Baseline Performance
        baseline_row = group[group['hint_set'] == 'baseline']
        if baseline_row.empty: 
            results.append({'query_id': qid_str, 'recommended_hint': 'NATIVE', 'gain': 1.0, 'status': 'SKIPPED'})
            continue
        source_baseline_time = baseline_row.iloc[0]['execution_time_s']
        
        # 2. Evaluate Candidates
        candidates = []
        for _, row in group.iterrows():
            h_name = row['hint_set']
            if h_name == 'baseline': continue
            
            source_hinted_time = row['execution_time_s']
            source_speedup = source_baseline_time / max(source_hinted_time, 0.001)
            
            # Skip candidates that were clearly worse on Source
            if source_speedup < 0.95:
                continue
                
            # 3. Get Plan Structure for Risk Assessment
            explain_json = get_mysql_explain(sql_file, HINT_DEFINITIONS[h_name])
            if not explain_json: continue
            
            ops = parse_explain_json(explain_json, query_id=qid_str, env_id='A_SATA')
            if not ops: continue
            
            # Enrich and Propagate
            df_ops = pd.DataFrame(ops)
            # Add table/workload metadata (crucial for zero-shot)
            df_ops = enrich_operators_batch(df_ops, benchmark_col='benchmark', env_col='env_id')
            enriched_ops = df_ops.to_dict('records')
            # For Source execution data, we scale self_time/total_work by total actual time
            total_est_cost = sum(op['est_cost'] for op in enriched_ops if op['depth'] == 0)
            if total_est_cost > 0:
                scale = source_hinted_time / total_est_cost
                for op in enriched_ops:
                    op['self_time'] = op.get('est_cost', 0) * scale # Proxy
                    op['actual_rows'] = op.get('est_rows', 1)  # Zero-shot assumption
            
            propagate_tree_features(enriched_ops)
            
            # 4. HALO v4 Risk Prediction
            op_risks = []
            n_hr = 0
            total_risk_val = 0.0
            
            for op in enriched_ops:
                mu, sigma = halo_instance._predict_operator_probabilistic(op, hw_pair)
                upper_bound = halo_instance.calibrator.compute_upper_bound(mu, sigma)
                base_thresh = halo_instance._get_adaptive_threshold(op['op_type'])
                
                is_hr = upper_bound > base_thresh
                if is_hr: n_hr += 1
                
                risk_val = max(0, mu + 0.5 * sigma)
                total_risk_val += risk_val
                
                op_risks.append(OperatorRisk(op['op_type'], op.get('table_alias',''), op['depth'], mu, sigma, upper_bound, 'HIGH' if is_hr else 'SAFE', is_hr))
                
            risk_probability = min(1.0, n_hr / max(len(enriched_ops), 1))
            penalty = 0.04 * total_risk_val / max(math.log(len(enriched_ops) + 1), 1)
            expected_gain = source_speedup * math.exp(-penalty)
            
            # DEBUG
            if qid_str == 'q10':
                print(f"DEBUG q10: hint={h_name}, src_speedup={source_speedup:.4f}, penalty={penalty:.4f}, total_risk={total_risk_val:.4f}, gain={expected_gain:.4f}")

            avg_upper = np.mean([r.upper_bound for r in op_risks]) if op_risks else 0
            
            candidates.append(HintCandidate(h_name, source_speedup, source_speedup / math.exp(avg_upper), len(enriched_ops), n_hr, total_risk_val, op_risks, expected_gain, risk_probability))

        # 5. Selection Logic (HALO-P Performance - NO PENALTY)
        recommended = {'query_id': qid_str, 'recommended_hint': 'NATIVE', 'gain': 1.0, 'status': 'SAFE'}
        if candidates:
            # Pick based on source_speedup instead of expected_gain
            candidates.sort(key=lambda x: x.source_speedup, reverse=True)
            best = candidates[0]
            
            if best.source_speedup > 1.01:
                recommended = {
                    'query_id': qid_str, 
                    'recommended_hint': best.hint_set, 
                    'gain': best.source_speedup,
                    'source_speedup': best.source_speedup,
                    'n_hr': best.n_high_risk,
                    'status': 'MAX_PERF'
                }
        results.append(recommended)
        
    return pd.DataFrame(results)

def write_hinted_sql(target_env, df_results):
    out_dir = f"{OUTPUT_BASE}_{target_env.lower()}"
    os.makedirs(out_dir, exist_ok=True)
    
    import re
    for _, row in df_results.iterrows():
        qid = row['query_id']
        h_key = row['recommended_hint']
        src_path = os.path.join(SQL_DIR, f"stats_{qid}.sql")
        dst_path = os.path.join(out_dir, f"stats_{qid}.sql")
        
        with open(src_path, 'r') as f:
            lines = f.readlines()
            
        if h_key != 'NATIVE':
            hint_str = HINT_DEFINITIONS[h_key]
            for i, line in enumerate(lines):
                 if "SELECT" in line.upper() and not line.startswith('--'):
                      if "SELECT COUNT(*)" in line.upper():
                          lines[i] = line.replace("SELECT COUNT(*)", f"SELECT {hint_str} COUNT(*)", 1)
                      else:
                          lines[i] = re.sub(r'SELECT', f'SELECT {hint_str}', line, count=1, flags=re.IGNORECASE)
                      break
        
        with open(dst_path, 'w') as f:
            f.writelines(lines)
    
    print(f"Generated {len(df_results)} queries in {out_dir}")

def main():
    halo = get_halo_v4()
    df_source = pd.read_parquet(SOURCE_DATA)
    
    for target in ['Xeon_NVMe', 'Xeon_SATA']:
        report = recommend_for_scenario(target, df_source, halo)
        
        print(f"\n=== Summary for {target} ===")
        print(report['recommended_hint'].value_counts())
        
        write_hinted_sql(target, report)
        
        # Save CSV report
        report.to_csv(f"/root/halo/results/stats_recommendation_{target.lower()}.csv", index=False)

if __name__ == '__main__':
    main()
