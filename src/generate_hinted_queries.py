# -*- coding: utf-8 -*-
"""
HALO v4 Recommendation Generator for Intel Xeon Silver 4310
===========================================================

Uses the HALO v4 Probabilistic MLP with Conformal Safety Gating
to recommend safe hints for both JOB and TPCH benchmarks.

Target Server: Intel(R) Xeon(R) Silver 4310
  - Xeon_NVMe (500K IOPS, 3.3 GHz)
  - Xeon_SATA (80K IOPS, 3.3 GHz)
"""

import os
import re
import json
import logging
import pandas as pd
from halo_framework import get_halo_v4

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('generate_xeon_v4')

# ── Paths ──
OUTPUT_DIR    = '/root/halo/results/hinted_queries_xeon_v4'
TPCH_SQL_DIR  = '/root/tpch-dbgen/queries_mysql'
JOB_SQL_DIR   = '/root/join-order-benchmark/queries'

# ── Hint definitions ──
HINT_DEFINITIONS = {
    'hint01': 'SET_VAR(optimizer_switch="block_nested_loop=off")',
    'hint02': 'SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off")',
    'hint04': 'SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")',
    'hint05': 'SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)',
}

def inject_hint(sql: str, hint_str: str) -> str:
    sql = re.sub(r'/\*\+.*?\*/', '', sql, flags=re.DOTALL).strip()
    match = re.search(r'\bSELECT\b', sql, flags=re.IGNORECASE)
    if not match: return sql
    pos = match.end()
    return sql[:pos] + f' /*+ {hint_str} */' + sql[pos:]

def load_sql(benchmark: str, query_id: str) -> str:
    if benchmark == 'TPCH':
        num = query_id.replace('tpch_q', '')
        path = os.path.join(TPCH_SQL_DIR, f'{num}.sql')
    else:
        path = os.path.join(JOB_SQL_DIR, f'{query_id}.sql')
    if not os.path.exists(path): return None
    with open(path, 'r') as f:
        content = f.read()
    lines = [l for l in content.splitlines() if not l.strip().startswith('--')]
    return '\n'.join(lines).strip()

def main():
    logger.info("Initializing HALO v4 Framework...")
    halo = get_halo_v4()
    
    TARGET_SCENARIOS = ['Xeon_NVMe', 'Xeon_SATA']
    BENCHMARKS = ['TPCH', 'JOB']
    
    results_summary = []
    
    # Clear existing results to ensure a fresh, clean set of 113/22 files
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for scenario in TARGET_SCENARIOS:
        for benchmark in BENCHMARKS:
            logger.info(f"\nProcessing scenario: {scenario} | benchmark: {benchmark}")
            storage_label = 'NVMe' if 'NVMe' in scenario else 'SATA'
            out_dir = os.path.join(OUTPUT_DIR, storage_label, benchmark)
            os.makedirs(out_dir, exist_ok=True)
            
            query_ids = sorted(halo.df_queries[halo.df_queries['benchmark'] == benchmark]['query_id'].unique())
            
            for qid in query_ids:
                rec = halo.recommend(qid, 'A_NVMe', scenario, policy='performance')
                sql = load_sql(benchmark, qid)
                if not sql: continue
                
                if rec.recommended_hint:
                    hint_str = HINT_DEFINITIONS.get(rec.recommended_hint)
                    hinted_sql = inject_hint(sql, hint_str)
                    status = f"✅ {rec.recommended_hint}"
                else:
                    hinted_sql = sql
                    status = "🛡️  NATIVE"
                
                # Standardize to EXACTLY one file per query
                out_path = os.path.join(out_dir, f"{qid}.sql")
                
                # Strip trailing semicolon and whitespace to prevent ';;'
                clean_sql = hinted_sql.strip().rstrip(';')
                
                header = (
                    f"-- HALO v4 Recommended SQL (Performance-Focused)\n"
                    f"-- Query     : {qid}\n"
                    f"-- Scenario  : {scenario}\n"
                    f"-- Mode      : HALO-P v4 (Power/Performance Mode)\n"
                    f"-- Hint      : {rec.recommended_hint or 'NATIVE'}\n"
                    f"-- Risk Level : {rec.risk_level}\n"
                    f"-- Reason    : {rec.reason}\n"
                    f"{'='*70}\n\n"
                )
                with open(out_path, 'w') as f:
                    f.write(header + clean_sql + ';\n')
                
                results_summary.append({
                    'scenario': scenario,
                    'benchmark': benchmark,
                    'query_id': qid,
                    'recommended_hint': rec.recommended_hint,
                    'risk_level': rec.risk_level,
                    'reason': rec.reason
                })
                
                logger.info(f"  {qid:<10} → {status:<10} | {rec.risk_level} | {rec.reason[:50]}...")

    # Summary table
    df_sum = pd.DataFrame(results_summary)
    print(f"\n{'='*80}")
    print(f"HALO v4 XEON RECOMMENDATION SUMMARY")
    print(f"{'='*80}")
    for (scen, bm), sub in df_sum.groupby(['scenario', 'benchmark']):
        native_count = (sub['recommended_hint'].isna()).sum()
        hinted_count = len(sub) - native_count
        print(f"[{scen}] {bm}: {len(sub)} queries | Hinted: {hinted_count} | Native: {native_count}")
        if hinted_count > 0:
            print(sub['recommended_hint'].value_counts().to_string())
    
    # Save to JSON for analysis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'recommendation_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n✅ All v4 recommendations saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
