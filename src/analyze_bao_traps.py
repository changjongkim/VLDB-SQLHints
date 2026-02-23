# -*- coding: utf-8 -*-
"""
Analysis of BAO's Regressions vs HALO's Decisions
================================================
Focus on: "Where did BAO fail, and did HALO save us?"
"""
import pandas as pd
import numpy as np
from halo_framework import HaloFramework

def analyze_failures():
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')
    
    halo = HaloFramework()
    halo.train(df_ops, df_q)
    
    # We focus on the most difficult transfers
    transfer_scenarios = [
        ('A_SATA', 'B_NVMe'),
        ('B_NVMe', 'A_SATA'),
        ('B_SATA', 'A_NVMe'),
        ('B_NVMe', 'B_SATA'),
    ]
    
    print(f"{'Scenario':<20} | {'Query':<10} | {'BAO Result':<12} | {'HALO Decision':<15} | {'Status'}")
    print("-" * 80)
    
    total_bao_regs = 0
    saved_by_halo = 0
    
    for src, tgt in transfer_scenarios:
        # Get HALO results for this transfer
        eval_result = halo.evaluate_transfer(src, tgt)
        
        # Calculate BAO manually for all queries in this transfer
        queries = set(df_q[df_q['env_id'] == src]['query_id'])
        for qid in sorted(queries):
            # 1. Get Target Baseline
            tgt_bl = halo._query_baselines.get((tgt, qid))
            if tgt_bl is None: continue
            
            # 2. BAO pick: best in source
            best_hs = None
            best_src_spd = 1.0
            src_bl = halo._query_baselines.get((src, qid), 1.0)
            
            src_hints = df_q[(df_q['env_id'] == src) & (df_q['query_id'] == qid) & (df_q['hint_set'] != 'baseline')]
            if len(src_hints) > 0:
                best_row = src_hints.loc[src_hints['execution_time_s'].idxmin()]
                best_src_spd = src_bl / max(best_row['execution_time_s'], 0.001)
                best_hs = best_row['hint_set']
            
            if best_hs and best_src_spd > 1.05:
                # BAO would apply this hint
                tgt_hint_row = df_q[(df_q['env_id'] == tgt) & (df_q['query_id'] == qid) & (df_q['hint_set'] == best_hs)]
                if len(tgt_hint_row) > 0:
                    actual_tgt_time = tgt_hint_row['execution_time_s'].min()
                    bao_actual_spd = tgt_bl / max(actual_tgt_time, 0.001)
                    
                    if bao_actual_spd < 0.95: # BAO REGRESSION!
                        total_bao_regs += 1
                        
                        # Now, what did HALO do?
                        halo_rec = halo.recommend(qid, src, tgt)
                        
                        status = ""
                        if halo_rec.recommended_hint is None:
                            saved_by_halo += 1
                            status = "✅ SAVED (Fallback)"
                        elif halo_rec.recommended_hint == best_hs:
                            status = "❌ FAILED (Both Regressed)"
                        else:
                            status = f"❓ OTHER ({halo_rec.recommended_hint})"
                            
                        print(f"{src+'->'+tgt:<20} | {qid:<10} | {bao_actual_spd:>10.2f}x | {str(halo_rec.recommended_hint):<15} | {status}")

    print("-" * 80)
    print(f"Total BAO Regressions identified: {total_bao_regs}")
    print(f"Saved by HALO (Native Fallback):  {saved_by_halo}")
    if total_bao_regs > 0:
        print(f"Rescue Rate:                      {(saved_by_halo/total_bao_regs)*100:.1f}%")

if __name__ == '__main__':
    analyze_failures()
