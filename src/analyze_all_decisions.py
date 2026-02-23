# -*- coding: utf-8 -*-
"""
HALO Decision Audit Report
==========================
Generates a full audit trail for every single query recommendation.
Helps verify the 'validity' of each decision and the underlying sigma logic.
"""
import pandas as pd
import numpy as np
import os
from halo_framework import HaloFramework

def audit_all_decisions():
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')
    
    halo = HaloFramework()
    halo.train(df_ops, df_q)
    
    envs = sorted(['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA'])
    all_rows = []
    
    print("Auditing all decisions across 16 transfers...")
    
    for src in envs:
        for tgt in envs:
            if src == tgt: continue
            
            recs = halo.recommend_all(src, tgt, policy='robust')
            
            for rec in recs:
                # Get actual speedup from ground truth
                tgt_bl = halo._query_baselines.get((tgt, rec.query_id))
                actual_spd = 1.0
                if rec.recommended_hint:
                    ht = df_q[(df_q['env_id'] == tgt) & (df_q['query_id'] == rec.query_id) & (df_q['hint_set'] == rec.recommended_hint)]['execution_time_s']
                    if len(ht) > 0 and tgt_bl:
                        actual_spd = tgt_bl / max(ht.min(), 0.001)
                
                # Analyze why it was native if it's a fallback
                high_risk_ops = []
                best_candidate_spd = 1.0
                if rec.native_fallback:
                    if rec.all_candidates:
                        best_c = max(rec.all_candidates, key=lambda x: x.source_speedup)
                        best_candidate_spd = best_c.source_speedup
                        high_risk_ops = [f"{o.op_type}({o.predicted_delta:+.2f})" for o in best_c.operator_details if o.risk_level == 'HIGH']
                
                all_rows.append({
                    'Xfer': f"{src}->{tgt}",
                    'Query': rec.query_id,
                    'Decision': rec.recommended_hint or 'NATIVE',
                    'Source_Spd': rec.all_candidates[0].source_speedup if rec.all_candidates else 1.0,
                    'Pred_Spd': rec.predicted_speedup,
                    'Actual_Spd': actual_spd,
                    'High_Risk_Ops': ", ".join(high_risk_ops[:3]),
                    'Status': 'Correct' if (actual_spd >= 0.95) else 'Regression',
                    'Reason': rec.reason
                })

    df = pd.DataFrame(all_rows)
    
    # Save detailed CSV for user to inspect
    df.to_csv('/root/halo/results/halo_full_audit.csv', index=False)
    
    # Summary of why we pick NATIVE
    native_cases = df[df['Decision'] == 'NATIVE']
    print(f"\nAudit Complete. Total decisions: {len(df)}")
    print(f"Total NATIVE decisions: {len(native_cases)}")
    
    print("\n--- Decision Validity Examples ---")
    
    # Example 1: Successful Safety Fallback (BAO would fail)
    saved = df[(df['Decision'] == 'NATIVE') & (df['Source_Spd'] > 1.2)]
    print("\n1. Safety Fallbacks (Avoided potential issues where Source Speedup was high):")
    print(saved[['Xfer', 'Query', 'Source_Spd', 'High_Risk_Ops']].head(10).to_string())
    
    # Example 2: Conservative misses (Native but source speedup was huge)
    misses = df[(df['Decision'] == 'NATIVE') & (df['Source_Spd'] > 2.0)]
    print("\n2. Conservative Losses (We stayed Native even though Source was 2x+):")
    print(misses[['Xfer', 'Query', 'Source_Spd', 'High_Risk_Ops']].head(5).to_string())
    
    # Example 3: Good recommendations
    wins = df[(df['Decision'] != 'NATIVE') & (df['Actual_Spd'] > 1.5)]
    print("\n3. Confident Recommendations (Confirmed Speedups):")
    print(wins[['Xfer', 'Query', 'Actual_Spd', 'Pred_Spd']].head(5).to_string())

if __name__ == "__main__":
    audit_all_decisions()
