import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
import math
import re
from pathlib import Path
from hw_registry import compute_hw_features

# Base directories
BASE_DIR = "/root/halo"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "results", "sigma_model_v3.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "hinted_queries_xeon_multi_source")

TPCH_SQL_DIR  = '/root/tpch-dbgen/queries_mysql'
JOB_SQL_DIR   = '/root/join-order-benchmark/queries'

HINT_DEFINITIONS = {
    'hint01': 'SET_VAR(optimizer_switch="block_nested_loop=off")',
    'hint02': 'SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off")',
    'hint04': 'SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")',
    'hint05': 'SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)',
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def engineer_features(op, hw):
    feats = op.copy()
    feats.update(hw)
    return feats

def inject_hint(sql: str, hint_str: str) -> str:
    """Insert /*+ HINT */ right after the first SELECT."""
    sql = re.sub(r'/\*\+.*?\*/', '', sql, flags=re.DOTALL).strip()
    match = re.search(r'\bSELECT\b', sql, flags=re.IGNORECASE)
    if not match: return sql
    pos = match.end()
    return sql[:pos] + f' /*+ {hint_str} */' + sql[pos:]

def predict_sigma_multi_hset(regressor, classifier, feature_cols, hsets_ops, hw, adaptive_thresholds):
    all_rows = []
    hset_boundaries = []
    op_types = []
    
    curr = 0
    for hset, ops in hsets_ops.items():
        for op in ops:
            feats = engineer_features(op, hw)
            all_rows.append([feats.get(c, 0.0) for c in feature_cols])
            op_types.append(op.get('op_type', 'UNKNOWN'))
        hset_boundaries.append((hset, curr, curr + len(ops)))
        curr += len(ops)
        
    if not all_rows: return {}
    X = np.array(all_rows, dtype=np.float32)
    
    subset_estimators = regressor.estimators_[::10] 
    tree_preds = np.array([tree.predict(X) for tree in subset_estimators])
    
    sigmas_mean = np.mean(tree_preds, axis=0)
    sigmas_std  = np.std(tree_preds, axis=0)
    is_high_risk_clf = classifier.predict(X) if classifier else np.zeros(len(all_rows))
    
    results = {}
    for hset, start, end in hset_boundaries:
        h_means = sigmas_mean[start:end]
        h_stds  = sigmas_std[start:end]
        h_clfs  = is_high_risk_clf[start:end]
        h_types = op_types[start:end]
        is_hr_thresh = [1 if (abs(h_means[i]) > adaptive_thresholds.get(h_types[i], 0.8)) else 0 for i in range(len(h_means))]
        is_high_risk = [(h_clfs[i] or is_hr_thresh[i]) for i in range(len(h_means))]
        results[hset] = {
            'means': h_means, 'stds': h_stds, 
            'risk_ratio': sum(is_high_risk) / len(is_high_risk)
        }
    return results

def calculate_u_score(src_speedup, means, stds):
    pessimistic_risks = means + 1.0 * stds
    total_penalty = float(pessimistic_risks[pessimistic_risks > 0].sum())
    return src_speedup * math.exp(-0.4 * total_penalty)

def load_sql(benchmark, qid):
    if benchmark == 'TPCH':
        num = qid.replace('tpch_q', '')
        path = os.path.join(TPCH_SQL_DIR, f'{num}.sql')
    else:  # JOB
        path = os.path.join(JOB_SQL_DIR, f'{qid}.sql')
    
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return f.read()

def main():
    df_ops = pd.read_parquet(os.path.join(DATA_DIR, 'unified_operators.parquet'))
    df_q   = pd.read_parquet(os.path.join(DATA_DIR, 'unified_queries.parquet'))
    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)
    
    reg_model = saved['model']
    feat_cols = saved['feature_cols']
    risk_clf  = saved.get('risk_classifier')
    adaptive_thresholds = saved.get('adaptive_thresholds', {})
    
    all_envs = df_q['env_id'].unique()
    target_scenarios = ['Xeon_NVMe', 'Xeon_SATA']
    benchmarks = ['TPCH', 'JOB']
    summary = []

    for scenario in target_scenarios:
        storage = 'NVMe' if 'NVMe' in scenario else 'SATA'
        env_max_risk = 0.15 if storage == 'SATA' else 0.35
        
        for bm in benchmarks:
            out_dir = os.path.join(OUTPUT_DIR, storage, bm)
            os.makedirs(out_dir, exist_ok=True)
            
            q_subset = df_q[df_q['benchmark'] == bm]
            qids = sorted(q_subset['query_id'].unique())
            logger.info(f"Processing {scenario} | {bm}...")
            
            for qid in qids:
                global_cands = {}
                for src_env in all_envs:
                    if src_env == scenario: continue
                    src_q = q_subset[(q_subset['env_id'] == src_env) & (q_subset['query_id'] == qid)]
                    if src_q.empty: continue
                    
                    base_time = src_q[src_q['hint_set'] == 'baseline']['execution_time_s'].min()
                    hint_data = src_q[src_q['hint_set'] != 'baseline']
                    if hint_data.empty: continue
                    
                    src_ops_df = df_ops[(df_ops['env_id'] == src_env) & (df_ops['query_id'] == qid)]
                    hw_feats = compute_hw_features(src_env, scenario)
                    
                    hsets_ops = {}
                    hsets_speedup = {}
                    for hset in hint_data['hint_set'].unique():
                        h_time = hint_data[hint_data['hint_set'] == hset]['execution_time_s'].min()
                        h_sp = base_time / max(h_time, 0.001)
                        if h_sp < 1.05: continue
                        h_ops = src_ops_df[src_ops_df['hint_set'] == hset].sort_values('tree_position').to_dict('records')
                        if h_ops:
                            hsets_ops[hset] = h_ops
                            hsets_speedup[hset] = h_sp
                    
                    if not hsets_ops: continue
                    preds = predict_sigma_multi_hset(reg_model, risk_clf, feat_cols, hsets_ops, hw_feats, adaptive_thresholds)
                    for hset, p in preds.items():
                        score = calculate_u_score(hsets_speedup[hset], p['means'], p['stds'])
                        if p['risk_ratio'] <= env_max_risk:
                            if hset not in global_cands or score > global_cands[hset]['score']:
                                global_cands[hset] = {'score': score, 'sp': hsets_speedup[hset], 'risk': p['risk_ratio'], 'from_env': src_env}

                rec_hint, reason = None, f"HALO-U: Uncertainty too high (Gate={env_max_risk:.0%})"
                if global_cands:
                    rec_hint = max(global_cands, key=lambda k: global_cands[k]['score'])
                    v = global_cands[rec_hint]
                    reason = f"HALO-U (Novelty): '{rec_hint}' (score={v['score']:.2f}, src_sp={v['sp']:.1f}x, risk={v['risk']:.0%}) from {v['from_env']}"
                
                sql = load_sql(bm, qid)
                if sql:
                    hint_label = rec_hint if rec_hint else 'native'
                    header = f"-- HALO Optimized\n-- Query: {qid}\n-- Scenario: {scenario}\n-- Reason: {reason}\n"
                    if rec_hint and rec_hint in HINT_DEFINITIONS:
                        sql_hinted = inject_hint(sql, HINT_DEFINITIONS[rec_hint])
                        final_sql = header + sql_hinted
                    else:
                        final_sql = header + sql
                        hint_label = 'native'
                    
                    filename = f"{qid}_{hint_label}.sql"
                    with open(os.path.join(out_dir, filename), 'w') as f:
                        f.write(final_sql)
                
                summary.append({'scenario': scenario, 'benchmark': bm, 'query_id': qid, 'hint': rec_hint or 'NATIVE', 'reason': reason})

    with open(os.path.join(OUTPUT_DIR, 'recommendation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Done!")

if __name__ == "__main__":
    main()
