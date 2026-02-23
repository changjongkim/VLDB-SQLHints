# -*- coding: utf-8 -*-
"""
JOB Hinted Query Generator:
Parses hint_progress.txt files from kcj_sqlhint to get speedup per hint,
then runs HALO-R sigma-model risk filter for Xeon Silver 4310 target,
and generates hinted SQL files for all 113 JOB queries.
"""

import os, re, json, pickle, math
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──
BASE_JOB  = '/root/kcj_sqlhint/server_b/SQL_hint/Original_Result_Logs/SATA/JOB'
JOB_SQL   = '/root/join-order-benchmark/queries'
MODEL_PATH= '/root/halo/results/sigma_model_v3.pkl'
DATA_DIR  = '/root/halo/data'
OUT_DIR   = '/root/halo/results/hinted_queries'

# Hint label → hint_progress.txt location & MySQL hint string
HINT_MAP = {
    'hint01': {
        'prog': os.path.join(BASE_JOB, 'Hint_01_single_hints/hint_progress.txt'),
        'sql':  'SET_VAR(optimizer_switch="block_nested_loop=off")',
    },
    'hint02': {
        'prog': os.path.join(BASE_JOB, 'Hint_02_table_specific/hint_progress.txt'),
        'sql':  'SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off")',
    },
    'hint04': {
        'prog': os.path.join(BASE_JOB, 'Hint_04_join_order_index/hint_progress.txt'),
        'sql':  'SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")',
    },
    'hint05': {
        'prog': os.path.join(BASE_JOB, 'Hint_05_index_strategy/hint_progress.txt'),
        'sql':  'SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)',
    },
}


def parse_progress(path: str) -> dict:
    """
    Parse hint_progress.txt → {query_id: best_time_sec}
    Format: [N/M] 1a_join_method_1 - SUCCESS - 1.680270879s
    The query_id is the first token that matches a JOB query pattern (digits+letters).
    """
    times = {}
    if not os.path.exists(path):
        return {}
    # Match any line: [N/M] <qid>_<anything> - SUCCESS - <time>s
    pat = re.compile(r'\[[\d/]+\]\s+([0-9]+[a-z]+)_\S*\s+-\s+SUCCESS\s+-\s+([\d.]+)s')
    with open(path, 'r') as f:
        for line in f:
            m = pat.search(line)
            if m:
                qid = m.group(1)  # e.g. "1a", "10b"
                t   = float(m.group(2))
                times.setdefault(qid, []).append(t)
    return {qid: min(ts) for qid, ts in times.items()}


def parse_baseline(path: str) -> dict:
    """
    Parse baseline_progress.txt → {query_id: time_sec}
    Format: [N/113] 1a.sql - SUCCESS - 1.730922003s
    """
    times = {}
    if not os.path.exists(path):
        return {}
    pat = re.compile(r'\[[\d/]+\]\s+(\w+)\.sql\s+-\s+SUCCESS\s+-\s+([\d.]+)s')
    with open(path, 'r') as f:
        for line in f:
            m = pat.search(line)
            if m:
                times[m.group(1)] = float(m.group(2))
    return times


def engineer_features(op: dict, hw: dict) -> dict:
    features = {}
    op_types = ['NL_INNER_JOIN','FILTER','TABLE_SCAN','INDEX_LOOKUP','HASH_JOIN',
                'AGGREGATE','SORT','NL_ANTIJOIN','NL_SEMIJOIN','INDEX_SCAN',
                'TEMP_TABLE_SCAN','STREAM','MATERIALIZE','GROUP_AGGREGATE','NL_LEFT_JOIN']
    ot = op.get('op_type','UNKNOWN')
    for o in op_types:
        features[f'is_{o}'] = 1.0 if ot==o else 0.0
    features['is_join']     = 1.0 if 'JOIN' in ot else 0.0
    features['is_scan']     = 1.0 if 'SCAN' in ot else 0.0
    features['is_index_op'] = 1.0 if 'INDEX' in ot else 0.0
    features['is_agg_sort'] = 1.0 if ot in ('AGGREGATE','SORT','GROUP_AGGREGATE') else 0.0

    actual_rows = max(op.get('actual_rows',1),1)
    est_rows    = max(op.get('est_rows',1),1)
    loops       = max(op.get('loops',1),1)
    est_cost    = max(op.get('est_cost',0.01),0.01)
    self_time   = max(op.get('self_time',0.001),0.001)
    total_work  = max(op.get('total_work',0.001),0.001)

    features['log_actual_rows'] = math.log(actual_rows)
    features['log_est_rows']    = math.log(est_rows)
    features['log_loops']       = math.log(loops)
    features['log_est_cost']    = math.log(est_cost)
    features['row_est_error']   = op.get('row_est_error',1.0)
    features['rows_per_loop']   = math.log(actual_rows/loops+1)
    features['log_self_time']   = math.log(self_time)
    features['log_total_work']  = math.log(total_work)
    features['cost_per_row']    = math.log(est_cost/est_rows+0.001)
    features['time_per_row']    = math.log(self_time/actual_rows+0.0001)
    features['selectivity']     = math.log(actual_rows/est_rows+0.001)
    features['depth']           = op.get('depth',0)
    features['num_children']    = op.get('num_children',0)

    sc = hw['storage_changed']; cc = hw['compute_changed']
    features['storage_changed'] = sc
    features['compute_changed'] = cc
    features['both_changed']    = sc*cc
    features['scan_x_storage']  = features['is_scan']*sc
    features['scan_x_compute']  = features['is_scan']*cc
    features['join_x_storage']  = features['is_join']*sc
    features['join_x_compute']  = features['is_join']*cc
    features['index_x_storage'] = features['is_index_op']*sc
    features['index_x_compute'] = features['is_index_op']*cc
    features['agg_x_storage']   = features['is_agg_sort']*sc
    features['agg_x_compute']   = features['is_agg_sort']*cc
    features['work_x_storage']  = features['log_total_work']*sc
    features['work_x_compute']  = features['log_total_work']*cc
    features['rows_x_storage']  = features['log_actual_rows']*sc
    features['rows_x_compute']  = features['log_actual_rows']*cc
    return features


def predict_sigma_batch(model, feat_cols, ops, hw):
    rows = [[engineer_features(op, hw).get(c,0.0) for c in feat_cols] for op in ops]
    return model.predict(np.array(rows, dtype=np.float32))


def halo_r(hint_speedups, ops_by_hint, model, feat_cols, hw,
           THR=0.8, MAX_RISK_RATIO=0.3):
    """Return (best_hint, reason)."""
    candidates = {}
    for hs, speedup in hint_speedups.items():
        ops = ops_by_hint.get(hs, [])
        if ops:
            sigmas = predict_sigma_batch(model, feat_cols, ops, hw)
            n_total = len(sigmas)
            n_risk  = int((sigmas > THR).sum())
            ratio   = n_risk / max(n_total,1)
        else:
            ratio = 0.0; n_risk = 0
        candidates[hs] = {'speedup': speedup, 'risk_ratio': ratio, 'n_risk': n_risk}

    # Primary: risk <= 30% AND speedup >= 1.0
    safe = {hs:v for hs,v in candidates.items()
            if v['risk_ratio'] <= MAX_RISK_RATIO and v['speedup'] >= 1.0}
    if safe:
        best = max(safe, key=lambda hs: safe[hs]['speedup'])
        v = safe[best]
        return best, (f"HALO-R: '{best}' (src={v['speedup']:.2f}x, "
                      f"risk={v['n_risk']}/{len(ops_by_hint.get(best,[]))}, "
                      f"ratio={v['risk_ratio']:.0%})")

    # Fallback: lowest risk with speedup >= 1.0
    pos = {hs:v for hs,v in candidates.items() if v['speedup'] >= 1.0}
    if pos:
        safest = min(pos, key=lambda hs: pos[hs]['risk_ratio'])
        v = pos[safest]
        if v['risk_ratio'] <= 0.5:
            return safest, (f"HALO-R: '{safest}' CAUTIOUS "
                            f"(src={v['speedup']:.2f}x, ratio={v['risk_ratio']:.0%})")

    all_info = ', '.join(f"{hs}({v['risk_ratio']:.0%})" for hs,v in candidates.items())
    return None, f"HALO-R: all too risky → NATIVE [{all_info}]"


def inject_hint(sql, hint_str):
    sql = re.sub(r'/\*\+.*?\*/', '', sql, flags=re.DOTALL).strip()
    m = re.search(r'\bSELECT\b', sql, re.IGNORECASE)
    if not m: return sql
    pos = m.end()
    return sql[:pos] + f' /*+ {hint_str} */' + sql[pos:]


def main():
    logger.info("Loading σ model and operator data...")
    with open(MODEL_PATH,'rb') as f:
        saved = pickle.load(f)
    model, feat_cols = saved['model'], saved['feature_cols']

    df_ops = pd.read_parquet(os.path.join(DATA_DIR,'unified_operators.parquet'))
    job_ops_src = df_ops[(df_ops['env_id']=='A_NVMe') & (df_ops['benchmark']=='JOB')]

    # Parse baseline times (Server B SATA, used as source reference for JOB hints)
    baseline_times = parse_baseline(os.path.join(BASE_JOB,'baseline_progress.txt'))
    logger.info(f"Parsed {len(baseline_times)} JOB baseline query times")

    # Parse hint times per hint set
    hint_times = {}
    for hs, info in HINT_MAP.items():
        hint_times[hs] = parse_progress(info['prog'])
        logger.info(f"  {hs}: {len(hint_times[hs])} queries")

    TARGET_SCENARIOS = {
        'Xeon_NVMe': {'storage_changed':0,'compute_changed':1},
        'Xeon_SATA': {'storage_changed':1,'compute_changed':1},
    }

    results = []
    for scenario, hw in TARGET_SCENARIOS.items():
        storage = 'NVMe' if hw['storage_changed']==0 else 'SATA'
        out_dir = os.path.join(OUT_DIR, storage, 'JOB')
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"\n=== {scenario}: Xeon Silver 4310 ({storage}) ===")

        all_qids = sorted(baseline_times.keys())
        for qid in all_qids:
            base_t = baseline_times[qid]

            # Speedup per hint set for this query
            speedups = {}
            for hs in HINT_MAP:
                t = hint_times[hs].get(qid)
                if t is not None:
                    speedups[hs] = base_t / max(t, 0.001)

            if not speedups:
                continue

            # Baseline operators from source (A_NVMe)
            ops = job_ops_src[job_ops_src['query_id']==qid].to_dict('records')
            # For JOB, hint_set column is only 'baseline' — use same ops for all hints
            ops_by_hint = {hs: ops for hs in speedups}

            best_hint, reason = halo_r(speedups, ops_by_hint, model, feat_cols, hw)

            # Load SQL
            sql_path = os.path.join(JOB_SQL, f'{qid}.sql')
            if not os.path.exists(sql_path):
                logger.warning(f"  SQL not found: {sql_path}")
                continue
            with open(sql_path) as f:
                sql = '\n'.join(l for l in f.read().splitlines()
                               if not l.strip().startswith('--')).strip()

            # Generate hinted SQL
            if best_hint:
                hint_str = HINT_MAP[best_hint]['sql']
                out_sql  = inject_hint(sql, hint_str)
                fname    = f"{qid}_{best_hint}.sql"
                status   = f"✅ {best_hint}"
            else:
                hint_str = None
                out_sql  = sql
                fname    = f"{qid}_native.sql"
                status   = "🛡️  NATIVE"

            header = (
                f"-- HALO Recommended SQL (JOB)\n"
                f"-- Query     : {qid}\n"
                f"-- Target    : Intel Xeon Silver 4310 ({storage})\n"
                f"-- Hint      : {best_hint or 'NATIVE'}\n"
                f"-- Reason    : {reason}\n"
                f"-- Src spdup : {max(speedups.values()):.2f}x (best available hint)\n"
                f"--\n\n"
            )
            with open(os.path.join(out_dir, fname),'w') as f:
                f.write(header + out_sql + ';\n')

            logger.info(f"  {qid:6s} → {status:12s} | {reason[:70]}")
            results.append({
                'scenario': scenario, 'benchmark': 'JOB', 'query_id': qid,
                'hint': best_hint or 'NATIVE', 'reason': reason,
                'baseline_s': round(base_t,3),
            })

    # Summary
    df = pd.DataFrame(results)
    print(f"\n{'='*60}\nJOB Recommendation Summary (Xeon Silver 4310)\n{'='*60}")
    for scen in df['scenario'].unique():
        sub = df[df['scenario']==scen]
        print(f"\n  [{scen}]: {len(sub)} queries | "
              f"Hinted: {(sub['hint']!='NATIVE').sum()} | "
              f"Native: {(sub['hint']=='NATIVE').sum()}")
        print(sub['hint'].value_counts().to_string())

    # Append to existing recommendation_summary.json
    summary_path = os.path.join(OUT_DIR,'recommendation_summary.json')
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
    with open(summary_path,'w') as f:
        json.dump(existing + results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ JOB SQL files saved to {OUT_DIR}/{{NVMe,SATA}}/JOB/")


if __name__ == '__main__':
    main()
