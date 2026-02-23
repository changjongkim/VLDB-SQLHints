# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_cpu_sensitivity():
    # Load data
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    
    # Filter for B_NVMe and A_NVMe to isolate CPU difference (Storage is same: NVMe)
    # B_NVMe: EPYC 7713 (256MB L3)
    # A_NVMe: i9-12900K (30MB L3)
    
    e1 = 'B_NVMe' # Server-grade
    e2 = 'A_NVMe' # Desktop-grade
    
    ops1 = df_ops[df_ops['env_id'] == e1].copy()
    ops2 = df_ops[df_ops['env_id'] == e2].copy()
    
    # Matching logic (simplified for quick analysis)
    # Match by key columns
    match_cols = ['query_id', 'hint_set', 'op_type', 'tree_position', 'benchmark']
    merged = pd.merge(ops1, ops2, on=match_cols, suffixes=('_B', '_A'))
    
    # Calculate Sigma (performance delta)
    # Positive delta means it got SLOWER on A (Desktop)
    merged['sigma'] = np.log(merged['self_time_A'].clip(0.001) / merged['self_time_B'].clip(0.001))
    
    # 1. Distribution by Operator Type
    target_ops = ['NL_INNER_JOIN', 'HASH_JOIN', 'SORT', 'TABLE_SCAN', 'INDEX_LOOKUP', 'FILTER']
    plot_data = merged[merged['op_type'].isin(target_ops)].copy()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='op_type', y='sigma', data=plot_data, order=target_ops, palette='viridis')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title('Performance Delta: B_NVMe (Server) -> A_NVMe (Desktop)\nPositive = Slower on Desktop', fontsize=14)
    plt.ylabel('Sigma (Log Ratio of Self Time)', fontsize=12)
    plt.xlabel('Operator Type', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    output_path = '/root/halo/results/figures/cpu_architecture_sensitivity.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sensitivity plot to {output_path}")

    # 2. Case Study Report in Text
    print("\n=== CPU Architecture Sensitivity (B_NVMe -> A_NVMe) ===")
    summary = plot_data.groupby('op_type')['sigma'].agg(['mean', 'median', 'std', 'count'])
    print(summary.sort_values('median', ascending=False))
    
    # 3. High Impact Query Analysis (e.g., JOB 11a)
    query_id = '11a'
    q_data = merged[merged['query_id'] == query_id].copy()
    if not q_data.empty:
        print(f"\n--- Case Study: Query {query_id} ---")
        # Sum of self times
        total_B = q_data['self_time_B'].sum()
        total_A = q_data['self_time_A'].sum()
        print(f"Total Self Time: B_NVMe={total_B:.3f}s, A_NVMe={total_A:.3f}s")
        print(f"Ratio (A/B): {total_A/total_B:.2f}x")
        
        print("\nMajor Operators in 11a:")
        q_ops = q_data.sort_values('self_time_B', ascending=False).head(5)
        for _, row in q_ops.iterrows():
            print(f"  {row['op_type']:<15} | B: {row['self_time_B']:.3f}s | A: {row['self_time_A']:.3f}s | Ratio: {row['self_time_A']/row['self_time_B']:.2f}x")

if __name__ == "__main__":
    analyze_cpu_sensitivity()
