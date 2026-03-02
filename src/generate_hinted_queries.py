# -*- coding: utf-8 -*-
"""
HALO v4 Recommendation Generator for Intel Xeon Silver 4310
===========================================================

Uses the HALO v4 Probabilistic MLP with Conformal Safety Gating
to recommend safe hints for both JOB and TPCH benchmarks.

IMPORTANT: JOB and TPC-H use fundamentally different hint mechanisms.
  - JOB: Hints are optimizer-switch comments injected as /*+ ... */
  - TPC-H: Hints modify table-level SQL (FORCE INDEX, JOIN_ORDER, etc.)
           These CANNOT be represented as simple comment injections.
           Instead, we reference the actual best-performing variant SQL
           from the experiment data directory.

Target Server: Intel(R) Xeon(R) Silver 4310
  - Xeon_NVMe (500K IOPS, 3.3 GHz)
  - Xeon_SATA (80K IOPS, 3.3 GHz)
"""

import os
import re
import json
import glob
import logging
import pandas as pd
from halo_framework import get_halo_v4

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('generate_xeon_v4')

# ── Paths ──
OUTPUT_DIR    = '/root/halo/results/hinted_queries_xeon_v4'
TPCH_SQL_DIR  = '/root/tpch-dbgen/queries_mysql'
JOB_SQL_DIR   = '/root/join-order-benchmark/queries'

# ── TPC-H experiment variant directories ──
# These contain the actual SQL files used during training/evaluation.
# Each hint set has a different directory with per-query, per-variant files.
TPCH_HINT_DIRS = {
    'hint01': '/root/tpch-dbgen/queries_with_hints/Hint_01_global',
    'hint02': '/root/tpch-dbgen/queries_with_hints/Hint_02_table_specific',
    'hint04': '/root/tpch-dbgen/queries_with_hints/Hint_04_join_order',
    'hint05': '/root/tpch-dbgen/queries_with_hints/Hint_05_index',
}

# ── TPC-H experiment result directories (for finding best variant) ──
TPCH_RESULT_DIRS = {
    'hint01': '/root/tpch-dbgen/NVMe_result/results/hint01_cold/benchmark_progress.log',
    'hint02': '/root/tpch-dbgen/NVMe_result/results/hint02_cold/benchmark_progress.log',
    'hint04': '/root/tpch-dbgen/NVMe_result/results/hint04_cold/benchmark_progress.log',
    'hint05': '/root/tpch-dbgen/NVMe_result/results/hint05_cold/benchmark_progress.log',
}

# ── JOB Hint definitions (optimizer-switch style, works via /*+ ... */) ──
JOB_HINT_DEFINITIONS = {
    'hint01': 'SET_VAR(optimizer_switch="block_nested_loop=off")',
    'hint02': 'SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off")',
    'hint04': 'SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")',
    'hint05': 'SET_VAR(optimizer_switch="block_nested_loop=off")',
}

def inject_hint(sql: str, hint_str: str) -> str:
    """Inject optimizer hint comment for JOB queries."""
    sql = re.sub(r'/\*\+.*?\*/', '', sql, flags=re.DOTALL).strip()
    match = re.search(r'\bSELECT\b', sql, flags=re.IGNORECASE)
    if not match: return sql
    pos = match.end()
    return sql[:pos] + f' /*+ {hint_str} */' + sql[pos:]

def load_sql(benchmark: str, query_id: str) -> str:
    """Load original (unhinted) SQL for a query."""
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

def parse_tpch_progress_log(log_path):
    """Parse TPC-H benchmark progress log to find exec times per variant."""
    results = {}
    if not os.path.exists(log_path):
        return results
    with open(log_path, 'r') as f:
        current_query = None
        for line in f:
            # Match: [1/154] Query 3_hint1_1 - ...
            qm = re.search(r'Query (\S+)\s+-', line)
            if qm:
                current_query = qm.group(1)
            # Match: ✓ SUCCESS - 353.893284489s
            tm = re.search(r'SUCCESS - ([\d.]+)s', line)
            if tm and current_query:
                t = float(tm.group(1))
                results[current_query] = t
                current_query = None
    return results

def find_best_tpch_variant(query_num, hint_set):
    """Find the best-performing variant SQL file for a TPC-H query and hint set.
    
    Returns the path to the best variant SQL file, or None if not found.
    """
    log_path = TPCH_RESULT_DIRS.get(hint_set)
    hint_dir = TPCH_HINT_DIRS.get(hint_set)
    
    if not log_path or not hint_dir:
        return None
    
    # Parse experiment results to find fastest variant
    variant_times = parse_tpch_progress_log(log_path)
    
    # Filter to variants for this query number
    # Variant names look like: 3_hint1_1, 3_hint1_2, etc.
    hint_num = hint_set.replace('hint0', 'hint').replace('hint', 'hint')  # hint01 -> hint1
    # Normalize: hint01 -> hint1, hint02 -> hint2, hint04 -> hint4, hint05 -> hint5
    hint_suffix = hint_set.replace('hint0', 'hint')  # hint01 -> hint1
    
    prefix = f"{query_num}_{hint_suffix}_"
    matching = {k: v for k, v in variant_times.items() if k.startswith(prefix)}
    
    if not matching:
        return None
    
    # Find the fastest variant
    best_variant = min(matching, key=matching.get)
    best_time = matching[best_variant]
    
    # Find corresponding SQL file
    sql_path = os.path.join(hint_dir, f"{best_variant}.sql")
    if os.path.exists(sql_path):
        logger.info(f"    Best variant for q{query_num}/{hint_set}: {best_variant} ({best_time:.1f}s)")
        return sql_path
    
    return None

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
                
                hinted_sql = None
                
                if rec.recommended_hint:
                    if benchmark == 'JOB':
                        # ── JOB: Use optimizer-switch injection (/*+ ... */) ──
                        hint_str = JOB_HINT_DEFINITIONS.get(rec.recommended_hint)
                        if hint_str:
                            hinted_sql = inject_hint(sql, hint_str)
                        else:
                            hinted_sql = sql
                    elif benchmark == 'TPCH':
                        # ── TPC-H: Use actual best-performing variant SQL from experiments ──
                        query_num = qid.replace('tpch_q', '')
                        variant_path = find_best_tpch_variant(query_num, rec.recommended_hint)
                        if variant_path:
                            with open(variant_path, 'r') as f:
                                variant_content = f.read()
                            # Strip comment lines from experiment files
                            lines = [l for l in variant_content.splitlines() if not l.strip().startswith('--')]
                            hinted_sql = '\n'.join(lines).strip()
                            logger.info(f"    TPC-H q{query_num}: Using experiment variant from {os.path.basename(variant_path)}")
                        else:
                            # Fallback: If no experiment variant found, use original SQL
                            logger.warning(f"    TPC-H q{query_num}: No experiment variant found for {rec.recommended_hint}, using NATIVE")
                            hinted_sql = sql
                            rec.recommended_hint = None
                            rec.risk_level = 'SAFE'
                            rec.reason = f"Fallback: No experiment variant available for {rec.recommended_hint}"
                    
                    status = f"✅ {rec.recommended_hint}" if rec.recommended_hint else "🛡️  NATIVE"
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
