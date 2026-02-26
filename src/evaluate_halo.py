import pandas as pd
import json
from halo_framework import HaloFramework

# Load data
df_o = pd.read_parquet('/root/halo/data/unified_operators.parquet')
df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')

halo = HaloFramework()
halo.train(df_o, df_q)

# ═══════════════════════════════════════════════════════════
#  Experiment: Full HALO-G vs HALO-R across ALL 12 transfers
# ═══════════════════════════════════════════════════════════
print('='*80)
print('  HALO-G vs HALO-R: Full Cross-Environment Evaluation')
print('='*80)

envs = ['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA']
all_g_reg = 0
all_r_reg = 0
all_rescues = 0

rows = []
for src in envs:
    for tgt in envs:
        if src == tgt:
            continue
        try:
            comp = halo.compare_policies(src, tgt)
            g = comp['halo_g']
            r = comp['halo_r']
            all_g_reg += g['n_regressions']
            all_r_reg += r['n_regressions']
            all_rescues += comp['rescue_count']
            rows.append({
                'transfer': f'{src} -> {tgt}',
                'n_queries': g['n_queries'],
                'G_hinted': g['n_hinted'], 'G_reg': g['n_regressions'], 'G_avg': g['avg_speedup'],
                'R_hinted': r['n_hinted'], 'R_reg': r['n_regressions'], 'R_avg': r['avg_speedup'],
                'rescues': comp['rescue_count'],
            })
        except Exception as e:
            print(f"Error evaluating {src} -> {tgt}: {e}")

# Print table
print(f"{'Transfer':<22} {'Qs':>4} {'G_hint':>6} {'G_reg':>5} {'G_avg':>7} {'R_hint':>6} {'R_reg':>5} {'R_avg':>7} {'Rescue':>7}")
print('-'*75)
for r in rows:
    print(f"{r['transfer']:<22} {r['n_queries']:>4} {r['G_hinted']:>6} {r['G_reg']:>5} {r['G_avg']:>7.3f} {r['R_hinted']:>6} {r['R_reg']:>5} {r['R_avg']:>7.3f} {r['rescues']:>7}")

print('-'*75)
print(f"{'TOTAL':<22} {'':<4} {'':<6} {all_g_reg:>5} {'':<7} {'':<6} {all_r_reg:>5} {'':<7} {all_rescues:>7}")
print()
if all_g_reg > 0:
    print(f"Regression Reduction:  {(1 - all_r_reg/all_g_reg)*100:.1f}%")
print(f"Total Rescues (G failed, R saved): {all_rescues}")

# ═══════════════════════════════════════════════════════════
#  Notable Rescue Cases
# ═══════════════════════════════════════════════════════════
print()
print('='*80)
print('  Notable Rescue Cases (HALO-G regressed, HALO-R saved)')
print('='*80)
for src in envs:
    for tgt in envs:
        if src == tgt: continue
        comp = halo.compare_policies(src, tgt)
        for rescue in comp['rescues']:
            print(f"  {src} -> {tgt} | {rescue['query_id']:<12} | "
                  f"G: {rescue['halo_g_hint']} = {rescue['halo_g_actual']:.2f}x | "
                  f"R: {rescue['halo_r_hint']} = {rescue['halo_r_actual']:.2f}x")
