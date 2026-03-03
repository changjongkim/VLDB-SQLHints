#!/usr/bin/env python3
"""
Zero-shot HALO Hint Generator for STATS Benchmark
Uses EXPLAIN FORMAT=JSON to obtain operator profiles without actual execution,
then passes them to HALO v4 for risk assessment and hint recommendation.
"""

import os
import json
import subprocess
import glob
from tqdm import tqdm
import pandas as pd

# Internal HALO modules
from halo_framework import get_halo_v4, HintCandidate, OperatorRisk
from explain_to_operators import parse_explain_json
from workload_metadata import enrich_operators_batch
from hw_registry import compute_hw_features
from tree_propagation import propagate_tree_features

MYSQL_CMD = "/usr/local/mysql-8.0.25/bin/mysql -u root -proot --socket=/tmp/mysql_3356.sock stats -BNe"

# Standard HALO hints for JOB/STATS (from original generate_hinted_queries.py)
HINT_DEFINITIONS = {
    'baseline': '',
    'hint01': '/*+ SET_VAR(optimizer_switch="block_nested_loop=off") */',
    'hint02': '/*+ BKA(v p c ph pl u b t) */',
    'hint04': '/*+ JOIN_ORDER(v, p, c, ph, pl, u, b, t) */',
    'hint05': '/*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */',
}

def get_explain_plan(sql_query: str, hint_comment: str = "") -> str:
    """Executes EXPLAIN FORMAT=JSON and returns the JSON string."""
    if hint_comment:
        if "SELECT COUNT(*)" in sql_query.upper():
            sql_query = sql_query.replace("SELECT COUNT(*)", f"SELECT {hint_comment} COUNT(*)", 1)
        elif "SELECT" in sql_query.upper():
            sql_query = sql_query.replace("SELECT", f"SELECT {hint_comment}", 1)
            
    clean_sql = sql_query.replace('"', '\\"')
    
    cmd = [
        "/usr/local/mysql-8.0.25/bin/mysql",
        "-u", "root",
        "-proot",
        "--socket=/tmp/mysql_3356.sock",
        "stats",
        "-BNe",
        f"EXPLAIN FORMAT=JSON {clean_sql}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return ""
            
        # Clean up mysql cli artifacts and unescaped newlines
        json_str = result.stdout.strip().replace(r'\n', '\n')
        
        # Output handles "mysql: [Warning] ..." gracefully
        if "{" in json_str:
            return json_str[json_str.find("{"):]
        return ""
    except Exception as e:
        return ""


def run_zero_shot_recommend(sql_path: str, qid: str, halo_instance):
    """
    1. Read original SQL
    2. Get EXPLAIN for baseline and all hints
    3. Parse into Operator Dicts -> Enrich -> Propagate
    4. Run HALO risk assessment
    """
    with open(sql_path, 'r') as f:
        # Skip comment lines
        lines = [l for l in f if not l.startswith('--')]
        original_sql = " ".join(lines).strip()

    # Step 1: Collect OP profiles
    all_ops_by_hint = {}
    valid_hints = []
    
    for h_name, h_comment in HINT_DEFINITIONS.items():
        explain_json = get_explain_plan(original_sql, h_comment)
        if not explain_json:
            continue
            
        # Hardcode source environment to A_NVMe for scale resemblance (NVMe)
        ops = parse_explain_json(explain_json, query_id=qid, env_id='A_NVMe')
        if not ops:
            continue
            
        # Enrich metadata (adds table_rows, total_dataset_gb, fits_in_buffer, etc.)
        df_ops = pd.DataFrame(ops)
        df_ops = enrich_operators_batch(df_ops, benchmark_col='benchmark', env_col='env_id')
        enriched_ops = df_ops.to_dict('records')
        
        # We need self_time>0 and actual_rows for propagation (defaults added in parser)
        propagate_tree_features(enriched_ops)
        
        all_ops_by_hint[h_name] = enriched_ops
        if h_name != 'baseline':
            valid_hints.append(h_name)

    baseline_ops = all_ops_by_hint.get('baseline')
    if not baseline_ops or not valid_hints:
         return {'query_id': qid, 'recommended_hint': None, 'reason': 'Failed to get EXPLAIN plans'}

    # -------------------------------------------------------------
    # Step 2: Zero-shot Risk Assessment using HALO model
    # -------------------------------------------------------------
    target_env = 'Xeon_NVMe'
    pair = compute_hw_features('A_NVMe', target_env)
    
    # We estimate baseline_time using root's total_work (est_cost) scaled
    baseline_time = max([op['est_cost'] for op in baseline_ops]) * 0.001 
    
    candidates = []
    for hs in valid_hints:
        hinted_ops = all_ops_by_hint[hs]
        hinted_cost = max([op['est_cost'] for op in hinted_ops])
        if hinted_cost == 0: continue
        
        hinted_time = hinted_cost * 0.001
        
        # Baseline speedup estimate
        source_speedup = baseline_time / max(hinted_time, 0.0001)
        
        # Diff operators
        diff_ops = [op for op in hinted_ops if op['est_cost'] > 0.1]
        
        op_risks = []
        n_hr = 0
        total_risk_val = 0.0
        
        for op in diff_ops:
            mu, sigma = halo_instance._predict_operator_probabilistic(op, pair)
            upper_bound = halo_instance.calibrator.compute_upper_bound(mu, sigma)
            base_thresh = halo_instance._get_adaptive_threshold(op['op_type'])
            
            is_hr = upper_bound > base_thresh
            if is_hr: n_hr += 1
            
            risk_val = max(0, mu + 0.5 * sigma)
            weight = 1.0 # simplified
            total_risk_val += risk_val * weight
            
            op_risks.append(OperatorRisk(op['op_type'], op.get('table_alias',''), op['depth'], mu, sigma, upper_bound, 'HIGH' if is_hr else 'SAFE', is_hr))

        import numpy as np
        import math
        avg_upper = np.mean([r.upper_bound for r in op_risks]) if op_risks else 0
        risk_probability = min(1.0, n_hr / max(len(op_risks), 1))
        
        # Penalty calculation
        penalty = 0.04 * total_risk_val / max(math.log(len(op_risks) + 1), 1)
        expected_gain = source_speedup * math.exp(-penalty)
        
        candidates.append(HintCandidate(hs, source_speedup, source_speedup / math.exp(avg_upper), len(diff_ops), n_hr, total_risk_val, op_risks, expected_gain, risk_probability))

    # -------------------------------------------------------------
    # Step 3: HALO-R Selection Logic
    # -------------------------------------------------------------
    if not candidates:
        return {'query_id': qid, 'recommended_hint': None, 'reason': 'No viable candidates'}
        
    candidates.sort(key=lambda x: x.expected_gain, reverse=True)
    best = candidates[0]
    
    # Target safety over raw performance (Robust mode)
    if best.n_high_risk > 0 or best.risk_probability > 0.3:
        safer = [c for c in candidates if c.n_high_risk == 0 and c.expected_gain > 1.05]
        if safer:
            best = safer[0]
            
    if best.expected_gain > 1.05:
         return {'query_id': qid, 'recommended_hint': best.hint_set, 'expected_gain': best.expected_gain, 'n_hr': best.n_high_risk}
         
    return {'query_id': qid, 'recommended_hint': None, 'reason': f'Gain too low ({best.expected_gain:.2f})'}


def main():
    halo = get_halo_v4()
    sql_files = sorted(glob.glob('/root/stats-benchmark/queries_mysql/stats_q*.sql'))
    
    print(f"Loaded HALO model. Processing {len(sql_files)} STATS queries (Zero-Shot)...")
    
    out_dir = '/root/stats-benchmark/queries_mysql_halo_xeon_nvme'
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    for fpath in tqdm(sql_files):
        qid = os.path.basename(fpath).split('_')[1].replace('.sql', '')
        res = run_zero_shot_recommend(fpath, qid, halo)
        results.append(res)
        
        # Write hinted query
        with open(fpath, 'r') as f:
            lines = f.readlines()
            
        hint_key = res.get('recommended_hint')
        if hint_key:
            hint_str = HINT_DEFINITIONS[hint_key]
            for i, line in enumerate(lines):
                 if "SELECT" in line.upper() and not line.startswith('--'):
                      if "SELECT COUNT(*)" in line.upper():
                          lines[i] = line.replace("SELECT COUNT(*)", f"SELECT {hint_str} COUNT(*)", 1)
                      else:
                          lines[i] = re.sub(r'SELECT', f'SELECT {hint_str}', line, count=1, flags=re.IGNORECASE)
                      break
                      
        out_path = os.path.join(out_dir, os.path.basename(fpath))
        with open(out_path, 'w') as f:
            f.writelines(lines)
        
    print(f"\nSaved all queries to {out_dir}")
    df = pd.DataFrame(results)
    
    # Summary of hints
    print("\nRecommendation Summary:")
    print(df['recommended_hint'].fillna('NATIVE').value_counts())
    
    # Save report
    df.to_csv(os.path.join(out_dir, 'halo_recommendation_report.csv'), index=False)

if __name__ == '__main__':
    import re
    main()
