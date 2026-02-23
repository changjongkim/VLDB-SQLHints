# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
from halo_framework import HaloFramework

def run_ablation():
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')
    
    halo = HaloFramework()
    halo.train(df_ops, df_q)
    
    all_envs = sorted(['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA'])
    
    results = []
    
    print(f"{'Scenario':<20} | {'Method':<12} | {'Reg':<5} | {'Speedup':<8}")
    print("-" * 55)
    
    for src in all_envs:
        for tgt in all_envs:
            if src == tgt: continue
            
            # 1. HALO-G (Greedy)
            g = halo.evaluate_transfer(src, tgt, policy='greedy')
            
            # 2. HALO-R (Robust)
            r = halo.evaluate_transfer(src, tgt, policy='robust')
            
            # 3. Base Baseline (Native always) - Actually it's 1.0 spd, 0 reg
            
            xfer = f"{src}->{tgt}"
            print(f"{xfer:<20} | {'HALO-G':<12} | {g['n_regressions']:<5} | {g['avg_speedup']:>8.3f}x")
            print(f"{xfer:<20} | {'HALO-R':<12} | {r['n_regressions']:<5} | {r['avg_speedup']:>8.3f}x")
            
            results.append({
                'transfer': xfer,
                'halo_g': g,
                'halo_r': r
            })
            
    # Calculate totals
    total_g_reg = sum(r['halo_g']['n_regressions'] for r in results)
    total_r_reg = sum(r['halo_r']['n_regressions'] for r in results)
    avg_g_spd = np.mean([r['halo_g']['avg_speedup'] for r in results])
    avg_r_spd = np.mean([r['halo_r']['avg_speedup'] for r in results])
    
    print("-" * 55)
    print(f"{'TOTAL':<20} | {'HALO-G':<12} | {total_g_reg:<5} | {avg_g_spd:>8.3f}x")
    print(f"{'TOTAL':<20} | {'HALO-R':<12} | {total_r_reg:<5} | {avg_r_spd:>8.3f}x")
    
    with open('/root/halo/results/ablation_quick.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_ablation()
