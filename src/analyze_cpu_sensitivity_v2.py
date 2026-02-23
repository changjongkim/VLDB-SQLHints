# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_cpu_sensitivity_v2():
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    
    # Environment IDs
    E_SERVER = 'B_NVMe' # EPYC
    E_DESKTOP = 'A_NVMe' # i9
    
    # Filter for the same hint_set to be fair
    # Usually hint05 or hint04 are the aggressive ones
    ops_server = df_ops[df_ops['env_id'] == E_SERVER]
    ops_desktop = df_ops[df_ops['env_id'] == E_DESKTOP]
    
    # Merge on identifier columns
    match_cols = ['benchmark', 'query_id', 'hint_set', 'tree_position', 'op_type']
    merged = pd.merge(ops_server, ops_desktop, on=match_cols, suffixes=('_SERVER', '_DESKTOP'))
    
    # Calculate relative speed (Server vs Desktop)
    # sigma = log(Desktop_Time / Server_Time)
    # Positive sigma = Server is FASTER (Desktop is slower)
    merged['sigma'] = np.log(merged['self_time_DESKTOP'].clip(0.001) / merged['self_time_SERVER'].clip(0.001))
    
    # 1. Visualization: Sigma Distribution by Operator
    # We want to see if some operators are "Server-friendly" (large positive sigma)
    # and others are "Desktop-friendly" (negative sigma)
    
    target_ops = ['NL_INNER_JOIN', 'HASH_JOIN', 'SORT', 'TABLE_SCAN', 'INDEX_LOOKUP', 'AGGREGATE']
    plot_df = merged[merged['op_type'].isin(target_ops)].copy()
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='op_type', y='sigma', data=plot_df, palette='Set2')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.title('CPU Architecture Sensitivity: Server (EPYC) vs Desktop (i9)\n(Log Ratio: Positive = Server is Faster, Negative = Desktop is Faster)', fontsize=12)
    plt.ylabel('Sigma (log(Desktop/Server))', fontsize=10)
    plt.grid(alpha=0.2)
    
    plt.savefig('/root/halo/results/figures/cpu_sensitivity_violin.png', dpi=300)
    
    # 2. Case Study: Why did HALO reject 11a?
    # JOB 11a often has massive joins.
    q11a = merged[merged['query_id'] == '11a'].copy()
    print("\n=== Analysis of Query 11a (B_NVMe -> A_NVMe) ===")
    if not q11a.empty:
        # Sort by total work impact
        q11a['impact'] = q11a['self_time_SERVER']
        top_ops = q11a.sort_values('impact', ascending=False).head(10)
        print(f"{'Op Type':<20} | {'Server(s)':<10} | {'Desktop(s)':<10} | {'Sigma':<6}")
        print("-" * 55)
        for _, row in top_ops.iterrows():
            print(f"{row['op_type']:<20} | {row['self_time_SERVER']:10.4f} | {row['self_time_DESKTOP']:10.4f} | {row['sigma']:+6.2f}")
    else:
        print("Query 11a matches not found in merged dataset.")

    # 3. Aggregate Table for Paper
    print("\n=== Operator Sensitivity Mean (Positive = Server is better) ===")
    stats = plot_df.groupby('op_type')['sigma'].mean().sort_values(ascending=False)
    print(stats)

if __name__ == "__main__":
    analyze_cpu_sensitivity_v2()
