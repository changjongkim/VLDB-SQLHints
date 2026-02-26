# -*- coding: utf-8 -*-
"""Quick evaluation of HALO-R v2 (3-Tier + Risk-Reward) vs HALO-G."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd, numpy as np
from halo_framework import HaloFramework

df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')

halo = HaloFramework()
halo.train(df_ops, df_q)

envs = sorted(['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA'])

print(f"{'Scenario':<20} | {'Method':<12} | {'Reg':<5} | {'Speedup':<8}", flush=True)
print('-' * 55, flush=True)

total_g_reg, total_r_reg = 0, 0
total_g_spd, total_r_spd = [], []
rescues = 0

for src in envs:
    for tgt in envs:
        if src == tgt: continue
        g = halo.evaluate_transfer(src, tgt, policy='greedy')
        r = halo.evaluate_transfer(src, tgt, policy='robust')
        
        xfer = f'{src}->{tgt}'
        print(f'{xfer:<20} | {"HALO-G":<12} | {g["n_regressions"]:<5} | {g["avg_speedup"]:>8.3f}x', flush=True)
        print(f'{xfer:<20} | {"HALO-Rv2":<12} | {r["n_regressions"]:<5} | {r["avg_speedup"]:>8.3f}x', flush=True)
        
        total_g_reg += g['n_regressions']
        total_r_reg += r['n_regressions']
        total_g_spd.append(g['avg_speedup'])
        total_r_spd.append(r['avg_speedup'])
        
        # Count rescues
        g_map = {d['query_id']: d for d in g['details']}
        r_map = {d['query_id']: d for d in r['details']}
        for qid in g_map:
            if qid in r_map:
                if g_map[qid]['actual'] < 0.95 and r_map[qid]['actual'] >= 0.95:
                    rescues += 1

print('-' * 55, flush=True)
print(f'{"TOTAL":<20} | {"HALO-G":<12} | {total_g_reg:<5} | {np.mean(total_g_spd):>8.3f}x', flush=True)
print(f'{"TOTAL":<20} | {"HALO-Rv2":<12} | {total_r_reg:<5} | {np.mean(total_r_spd):>8.3f}x', flush=True)
print(f'Regression Reduction: {(1 - total_r_reg/max(total_g_reg,1))*100:.0f}%', flush=True)
print(f'Rescue Cases: {rescues}', flush=True)
print(f'Speedup Retention: {np.mean(total_r_spd)/np.mean(total_g_spd)*100:.1f}%', flush=True)
