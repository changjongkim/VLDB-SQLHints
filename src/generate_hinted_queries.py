# -*- coding: utf-8 -*-
"""
HALO Query Generator for Intel Xeon Silver 4310 Target Server

Target: Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz
  - Compared to Server A (i9-12900K): compute_changed=1
  - Compared to Server B (EPYC-7713): compute_changed=1

This script:
  1. Loads unified_operators + unified_queries data
  2. Uses the sigma model (RF v3) to evaluate hint risk for target HW
  3. Applies HALO-R policy per query (Source: A_NVMe → Target: Xeon_NVMe or Xeon_SATA)
  4. Generates hinted SQL files for JOB and TPCH benchmarks

Output: /root/halo/results/hinted_queries/
  └─ NVMe/
  │   └─ JOB/  <query_id>.sql
  │   └─ TPCH/ <query_id>.sql
  └─ SATA/
      └─ JOB/  <query_id>.sql
      └─ TPCH/ <query_id>.sql
"""

import os
import re
import json
import pickle
import math
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from hw_registry import compute_hw_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──
DATA_DIR      = '/root/halo/data'
MODEL_PATH    = '/root/halo/results/sigma_model_v3.pkl'
OUTPUT_DIR    = '/root/halo/results/hinted_queries'
TPCH_SQL_DIR  = '/root/tpch-dbgen/queries_mysql'
JOB_SQL_DIR   = '/root/join-order-benchmark/queries'

# ── Hint definitions: MySQL optimizer_switch strings + join-order hints ──
HINT_DEFINITIONS = {
    'hint01': 'SET_VAR(optimizer_switch="block_nested_loop=off")',
    'hint02': 'SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off")',
    'hint04': 'SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")',
    'hint05': 'SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)',
}

# ── Simple hint injection (INSERT after first SELECT keyword) ──
def inject_hint(sql: str, hint_str: str) -> str:
    """Insert /*+ HINT */ right after the first SELECT."""
    # Remove existing hints if any
    sql = re.sub(r'/\*\+.*?\*/', '', sql, flags=re.DOTALL).strip()
    # Find first SELECT (case-insensitive)
    match = re.search(r'\bSELECT\b', sql, flags=re.IGNORECASE)
    if not match:
        return sql
    pos = match.end()
    return sql[:pos] + f' /*+ {hint_str} */' + sql[pos:]


def engineer_features(op: dict, hw: dict) -> dict:
    """Replicate v3 feature engineering for prediction (with quantitative HW + workload)."""
    features = {}
    op_types = ['NL_INNER_JOIN', 'FILTER', 'TABLE_SCAN', 'INDEX_LOOKUP',
                'HASH_JOIN', 'AGGREGATE', 'SORT', 'NL_ANTIJOIN',
                'NL_SEMIJOIN', 'INDEX_SCAN', 'TEMP_TABLE_SCAN',
                'STREAM', 'MATERIALIZE', 'GROUP_AGGREGATE', 'NL_LEFT_JOIN']
    ot = op.get('op_type', 'UNKNOWN')
    for o in op_types:
        features[f'is_{o}'] = 1.0 if ot == o else 0.0

    features['is_join']     = 1.0 if 'JOIN' in ot else 0.0
    features['is_scan']     = 1.0 if 'SCAN' in ot else 0.0
    features['is_index_op'] = 1.0 if 'INDEX' in ot else 0.0
    features['is_agg_sort'] = 1.0 if ot in ('AGGREGATE','SORT','GROUP_AGGREGATE') else 0.0

    actual_rows = max(op.get('actual_rows', 1), 1)
    est_rows    = max(op.get('est_rows', 1), 1)
    loops       = max(op.get('loops', 1), 1)
    est_cost    = max(op.get('est_cost', 0.01), 0.01)
    self_time   = max(op.get('self_time', 0.001), 0.001)
    total_work  = max(op.get('total_work', 0.001), 0.001)

    features['log_actual_rows'] = math.log(actual_rows)
    features['log_est_rows']    = math.log(est_rows)
    features['log_loops']       = math.log(loops)
    features['log_est_cost']    = math.log(est_cost)
    features['row_est_error']   = op.get('row_est_error', 1.0)
    features['rows_per_loop']   = math.log(actual_rows / loops + 1)
    features['log_self_time']   = math.log(self_time)
    features['log_total_work']  = math.log(total_work)
    features['cost_per_row']    = math.log(est_cost / est_rows + 0.001)
    features['time_per_row']    = math.log(self_time / actual_rows + 0.0001)
    features['selectivity']     = math.log(actual_rows / est_rows + 0.001)
    features['depth']           = op.get('depth', 0)
    features['num_children']    = op.get('num_children', 0)

    # Binary HW flags
    sc = hw['storage_changed']
    cc = hw['compute_changed']
    features['storage_changed'] = sc
    features['compute_changed'] = cc
    features['both_changed']    = hw.get('both_changed', sc * cc)

    # Quantitative HW ratios
    features['storage_speed_ratio']   = hw.get('storage_speed_ratio', 0.0)
    features['iops_ratio']            = hw.get('iops_ratio', 0.0)
    features['write_speed_ratio']     = hw.get('write_speed_ratio', 0.0)
    features['cpu_core_ratio']        = hw.get('cpu_core_ratio', 0.0)
    features['cpu_thread_ratio']      = hw.get('cpu_thread_ratio', 0.0)
    features['cpu_clock_ratio']       = hw.get('cpu_clock_ratio', 0.0)
    features['cpu_base_clock_ratio']  = hw.get('cpu_base_clock_ratio', 0.0)
    features['l3_cache_ratio']        = hw.get('l3_cache_ratio', 0.0)
    features['ram_ratio']             = hw.get('ram_ratio', 0.0)
    features['buffer_pool_ratio']     = hw.get('buffer_pool_ratio', 0.0)
    features['is_storage_downgrade']  = hw.get('is_storage_downgrade', 0)
    features['is_cpu_downgrade']      = hw.get('is_cpu_downgrade', 0)
    features['is_ram_downgrade']      = hw.get('is_ram_downgrade', 0)

    # Workload metadata
    features['log_table_rows']     = math.log(max(op.get('table_rows', 1), 1))
    features['log_table_size_mb']  = math.log(max(op.get('table_size_mb', 1), 1))
    features['n_indexes']          = op.get('n_indexes', 0)
    features['log_dataset_gb']     = math.log(max(op.get('dataset_total_gb', 0.1), 0.1))
    features['fits_in_buffer']     = op.get('fits_in_buffer', 0)

    # Operator × Binary HW interaction
    features['scan_x_storage']   = features['is_scan'] * sc
    features['scan_x_compute']   = features['is_scan'] * cc
    features['join_x_storage']   = features['is_join'] * sc
    features['join_x_compute']   = features['is_join'] * cc
    features['index_x_storage']  = features['is_index_op'] * sc
    features['index_x_compute']  = features['is_index_op'] * cc
    features['agg_x_storage']    = features['is_agg_sort'] * sc
    features['agg_x_compute']    = features['is_agg_sort'] * cc
    features['work_x_storage']   = features['log_total_work'] * sc
    features['work_x_compute']   = features['log_total_work'] * cc
    features['rows_x_storage']   = features['log_actual_rows'] * sc
    features['rows_x_compute']   = features['log_actual_rows'] * cc

    # Operator × Quantitative HW interaction
    iops_r  = hw.get('iops_ratio', 0.0)
    clock_r = hw.get('cpu_clock_ratio', 0.0)
    core_r  = hw.get('cpu_core_ratio', 0.0)
    features['scan_x_iops']           = features['is_scan'] * iops_r
    features['scan_x_storage_speed']  = features['is_scan'] * hw.get('storage_speed_ratio', 0.0)
    features['join_x_cpu_clock']      = features['is_join'] * clock_r
    features['join_x_cpu_cores']      = features['is_join'] * core_r
    features['agg_x_cpu_clock']       = features['is_agg_sort'] * clock_r
    features['index_x_iops']          = features['is_index_op'] * iops_r

    # Work × Quantitative HW interaction
    features['rows_x_iops']       = features['log_actual_rows'] * iops_r
    features['work_x_cpu_clock']  = features['log_total_work'] * clock_r
    features['cost_x_iops']       = features['log_est_cost'] * iops_r

    # Workload × HW cross-terms
    features['table_size_x_iops']           = features['log_table_size_mb'] * iops_r
    features['table_size_x_storage_speed']  = features['log_table_size_mb'] * hw.get('storage_speed_ratio', 0.0)
    features['table_rows_x_iops']           = features['log_table_rows'] * iops_r
    features['buffer_miss_x_iops']          = (1 - features['fits_in_buffer']) * iops_r
    features['dataset_x_ram']               = features['log_dataset_gb'] * hw.get('ram_ratio', 0.0)

    return features


def predict_sigma_batch(model, feature_cols: list, ops: list, hw: dict) -> np.ndarray:
    """Batch predict sigma for a list of operators (much faster than per-op calls)."""
    rows = []
    for op in ops:
        feats = engineer_features(op, hw)
        rows.append([feats.get(c, 0.0) for c in feature_cols])
    X = np.array(rows, dtype=np.float32)
    return model.predict(X)


def halo_r_recommend(query_ops: dict, source_baseline_time: float,
                     query_hint_times: dict, model, feat_cols: list, hw: dict,
                     HIGH_RISK_THRESHOLD=0.8, MAX_HIGH_RISK_RATIO=0.3):
    """
    HALO-R recommendation (batch prediction).
    For a new/unknown server, we use ratio-based risk threshold:
      reject only if >30% of operators are predicted high-risk.
    Returns (best_hint_set, reason) where best_hint_set may be None (native).
    """
    candidates = {}
    for hint_set, hint_ops in query_ops.items():
        if not hint_ops:
            continue
        sigmas      = predict_sigma_batch(model, feat_cols, hint_ops, hw)
        n_total     = len(sigmas)
        n_high_risk = int((sigmas > HIGH_RISK_THRESHOLD).sum())
        risk_ratio  = n_high_risk / max(n_total, 1)
        total_risk  = float(sigmas[sigmas > 0].sum())
        src_time    = query_hint_times.get(hint_set, source_baseline_time)
        src_speedup = source_baseline_time / max(src_time, 0.001)
        candidates[hint_set] = {
            'n_high_risk': n_high_risk,
            'risk_ratio':  risk_ratio,
            'total_risk':  total_risk,
            'src_speedup': src_speedup,
        }

    # Filter: keep hints where risk_ratio <= threshold AND source speedup > 1
    safe = {hs: v for hs, v in candidates.items()
            if v['risk_ratio'] <= MAX_HIGH_RISK_RATIO and v['src_speedup'] >= 1.0}

    if safe:
        best = max(safe, key=lambda hs: safe[hs]['src_speedup'])
        v = safe[best]
        reason = (f"HALO-R: '{best}' selected "
                  f"(src={v['src_speedup']:.2f}x, risk_ops={v['n_high_risk']}/{len(query_ops[best])}, "
                  f"risk_ratio={v['risk_ratio']:.0%})")
        return best, reason
    else:
        # Fallback: pick hint with lowest risk ratio and positive speedup
        positive = {hs: v for hs, v in candidates.items() if v['src_speedup'] >= 1.0}
        if positive:
            safest = min(positive, key=lambda hs: positive[hs]['risk_ratio'])
            v = positive[safest]
            if v['risk_ratio'] <= 0.5:  # at most 50% risky ops → cautious apply
                reason = (f"HALO-R: '{safest}' selected cautiously "
                          f"(src={v['src_speedup']:.2f}x, risk_ratio={v['risk_ratio']:.0%} HIGH)")
                return safest, reason
        all_hints = ", ".join(
            f"{hs}({v['risk_ratio']:.0%})" for hs, v in candidates.items())
        return None, f"HALO-R: All hints too risky. {all_hints} → NATIVE"


def load_sql(benchmark: str, query_id: str) -> str:
    """Load original SQL for a given query."""
    if benchmark == 'TPCH':
        # tpch_q9 → 9.sql
        num = query_id.replace('tpch_q', '')
        path = os.path.join(TPCH_SQL_DIR, f'{num}.sql')
    else:  # JOB
        # 10a → 10a.sql
        path = os.path.join(JOB_SQL_DIR, f'{query_id}.sql')

    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        content = f.read()
    # Strip comments
    lines = [l for l in content.splitlines() if not l.strip().startswith('--')]
    return '\n'.join(lines).strip()


def main():
    # ── Load data ──
    logger.info("Loading data...")
    df_ops  = pd.read_parquet(os.path.join(DATA_DIR, 'unified_operators.parquet'))
    df_q    = pd.read_parquet(os.path.join(DATA_DIR, 'unified_queries.parquet'))

    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)
    sig_model  = saved['model']
    feat_cols  = saved['feature_cols']
    logger.info(f"σ model loaded: {len(feat_cols)} features")

    # We will test generic recommendation to Target B (EPYC)
    # User requested to fix the target server to EPYC (Server B) for testing
    TARGET_SCENARIOS = ['B_NVMe', 'B_SATA']

    BENCHMARKS = ['TPCH', 'JOB']
    results_summary = []

    for scenario_name in TARGET_SCENARIOS:
        
        for benchmark in BENCHMARKS:
            # Dynamically select SOURCE_ENV based on where we have hint data
            SOURCE_ENV = 'A_NVMe' if benchmark == 'TPCH' else 'B_NVMe'
            
            # Use quantitative HW registry to dynamically calculate hardware transition
            hw = compute_hw_features(SOURCE_ENV, scenario_name)
            storage_label = 'NVMe' if 'NVMe' in scenario_name else 'SATA'
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Scenario: {SOURCE_ENV} → {scenario_name} (EPYC Server) | {benchmark}")

            out_dir = os.path.join(OUTPUT_DIR, storage_label, benchmark)
            os.makedirs(out_dir, exist_ok=True)

            # Get baseline times from source env
            src_base = df_q[
                (df_q['env_id'] == SOURCE_ENV) &
                (df_q['benchmark'] == benchmark) &
                (df_q['hint_set'] == 'baseline')
            ].groupby('query_id')['execution_time_s'].min().to_dict()

            # Get hint times from source env
            src_hint_times = df_q[
                (df_q['env_id'] == SOURCE_ENV) &
                (df_q['benchmark'] == benchmark) &
                (df_q['hint_set'] != 'baseline')
            ].groupby(['query_id', 'hint_set'])['execution_time_s'].min()

            # Get operators per query per hint set
            src_ops = df_ops[
                (df_ops['env_id'] == SOURCE_ENV) &
                (df_ops['benchmark'] == benchmark)
            ]

            query_ids = sorted(src_base.keys())
            logger.info(f"  [{benchmark}] Processing {len(query_ids)} queries...")

            for qid in query_ids:
                base_time = src_base.get(qid, 1.0)

                # Collect operators per hint set
                hint_ops = {}
                hint_times_for_q = {}
                for hs in ['hint01', 'hint02', 'hint04', 'hint05']:
                    ops = src_ops[
                        (src_ops['query_id'] == qid) &
                        (src_ops['hint_set'] == hs)
                    ].to_dict('records')
                    if ops:
                        hint_ops[hs] = ops
                    t = src_hint_times.get((qid, hs), None)
                    if t is not None:
                        hint_times_for_q[hs] = t

                if not hint_ops:
                    # No hint data available, skip
                    continue

                # HALO-R recommendation
                best_hint, reason = halo_r_recommend(
                    hint_ops, base_time, hint_times_for_q,
                    sig_model, feat_cols, hw
                )

                # Load original SQL
                sql = load_sql(benchmark, qid)
                if sql is None:
                    logger.warning(f"    SQL not found for {qid}")
                    continue

                # Generate output SQL
                if best_hint and best_hint in HINT_DEFINITIONS:
                    hint_str = HINT_DEFINITIONS[best_hint]
                    hinted_sql = inject_hint(sql, hint_str)
                    status = f"✅ {best_hint}"
                else:
                    hinted_sql = sql  # native, no hint
                    hint_str   = None
                    status = "🛡️  NATIVE"

                # Write SQL file
                sql_fname = f"{qid}_{'_'.join(filter(None,[best_hint]))}.sql" if best_hint else f"{qid}_native.sql"
                out_path  = os.path.join(out_dir, sql_fname)
                header    = (
                    f"-- HALO Recommended SQL\n"
                    f"-- Query     : {qid} ({benchmark})\n"
                    f"-- Scenario  : {SOURCE_ENV} → {scenario_name} (AMD EPYC Target)\n"
                    f"-- Hint      : {best_hint or 'NATIVE (no hint)'}\n"
                    f"-- Reason    : {reason}\n"
                    f"-- Hint Str  : {hint_str or 'N/A'}\n"
                    f"--\n\n"
                )
                with open(out_path, 'w') as f:
                    f.write(header + hinted_sql + ';\n')

                results_summary.append({
                    'scenario':   scenario_name,
                    'benchmark':  benchmark,
                    'query_id':   qid,
                    'hint':       best_hint or 'NATIVE',
                    'src_baseline_s': round(base_time, 3),
                    'reason':     reason,
                    'sql_file':   out_path,
                })
                logger.info(f"    {qid:10s} → {status:12s} | {reason[:70]}")

    # ── Save summary ──
    summary_path = os.path.join(OUTPUT_DIR, 'recommendation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # ── Print final table ──
    df_sum = pd.DataFrame(results_summary)
    print(f"\n{'='*70}")
    print("HALO Recommendation Summary (Target: AMD EPYC 7713)")
    print(f"{'='*70}")
    for scen in df_sum['scenario'].unique():
        for bm in df_sum['benchmark'].unique():
            sub = df_sum[(df_sum['scenario']==scen) & (df_sum['benchmark']==bm)]
            native_pct = (sub['hint']=='NATIVE').mean() * 100
            print(f"\n  [{scen}] {bm}: {len(sub)} queries | "
                  f"Hinted: {(sub['hint']!='NATIVE').sum()} | "
                  f"Native: {(sub['hint']=='NATIVE').sum()} ({native_pct:.0f}%)")
            hint_dist = sub['hint'].value_counts()
            for h, cnt in hint_dist.items():
                print(f"    {h:12s}: {cnt}")

    print(f"\n✅ SQL files saved to: {OUTPUT_DIR}")
    print(f"✅ Summary saved to:   {summary_path}")

if __name__ == '__main__':
    main()
