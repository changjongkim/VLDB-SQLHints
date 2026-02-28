
import pandas as pd
import json
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

def parse_progress_file(filepath):
    queries = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\] ([\w.]+)\.sql - .*? SUCCESS - ([\d.]+)s', line)
            if match:
                qid = match.group(1)
                time_s = float(match.group(2))
                queries.append({'query_id': qid, 'time': time_s})
    return pd.DataFrame(queries)

def main():
    FIGURES_DIR = '/root/halo/xeon_results/JOB/SATA/figures'
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load progress logs
    df_orig = parse_progress_file('/root/halo/xeon_results/JOB/SATA/original/progress.txt')
    df_halo = parse_progress_file('/root/halo/xeon_results/JOB/SATA/halo/progress.txt')
    
    # Merge and calculate speedup
    df = pd.merge(df_orig, df_halo, on='query_id', suffixes=('_orig', '_halo'))
    df['time_saved'] = df['time_orig'] - df['time_halo']
    df['speedup'] = df['time_orig'] / df['time_halo']
    
    # Load HALO recommendations
    with open('/root/halo/results/hinted_queries_xeon_v4/recommendation_summary.json') as f:
        recs = json.load(f)
    
    # Filter for Xeon_SATA and JOB
    recs_sata_job = [r for r in recs if r['scenario'] == 'Xeon_SATA' and r['benchmark'] == 'JOB']
    df_recs = pd.DataFrame(recs_sata_job)
    df_recs['recommended_hint'] = df_recs['recommended_hint'].fillna('NATIVE')
    
    # Merge performance with recommendations
    df = pd.merge(df, df_recs[['query_id', 'recommended_hint', 'risk_level', 'reason']], on='query_id')
    
    # --- 1. Hint Type Analysis ---
    print("=" * 60)
    print("1. PERFORMANCE IMPACT BY HINT TYPE")
    print("=" * 60)
    hint_stats = df.groupby('recommended_hint').agg(
        query_count=('query_id', 'count'),
        total_orig_time=('time_orig', 'sum'),
        total_halo_time=('time_halo', 'sum'),
        total_time_saved=('time_saved', 'sum')
    ).reset_index()
    hint_stats['effective_speedup'] = hint_stats['total_orig_time'] / hint_stats['total_halo_time']
    print(hint_stats.to_string(index=False))
    
    # --- 2. Risk Level Analysis (HALO-P Justification) ---
    print("\n" + "=" * 60)
    print("2. RISK LEVEL ANALYSIS (Justification of Performance Mode)")
    print("=" * 60)
    risk_stats = df.groupby('risk_level').agg(
        query_count=('query_id', 'count'),
        total_time_saved=('time_saved', 'sum'),
        avg_speedup=('speedup', 'mean')
    ).reset_index()
    print(risk_stats.to_string(index=False))
    
    # --- 3. Query Complexity Tier Analysis ---
    print("\n" + "=" * 60)
    print("3. BOTTLENECK RESOLUTION (By Original Time Tier)")
    print("=" * 60)
    def categorize_tier(t):
        if t < 10: return 'Fast (<10s)'
        if t < 100: return 'Medium (10s-100s)'
        return 'Heavy (>100s)'
    
    df['tier'] = df['time_orig'].apply(categorize_tier)
    tier_stats = df.groupby('tier').agg(
        query_count=('query_id', 'count'),
        orig_time_sum=('time_orig', 'sum'),
        halo_time_sum=('time_halo', 'sum'),
        time_saved_sum=('time_saved', 'sum')
    ).reset_index()
    tier_stats['tier_speedup'] = tier_stats['orig_time_sum'] / tier_stats['halo_time_sum']
    print(tier_stats.to_string(index=False))
    
    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot: Original vs Saved Time colored by Hint
    sns.scatterplot(data=df, x='time_orig', y='time_saved', hue='recommended_hint', size='speedup', sizes=(50, 300), alpha=0.7, ax=axes[0])
    axes[0].plot([0, 500], [0, 500], 'k--', alpha=0.3) # y=x line
    axes[0].set_title('Time Saved vs. Original Execution Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Original Time (s)', fontsize=12)
    axes[0].set_ylabel('Absolute Time Saved (s)', fontsize=12)
    
    # Bar plot: Cumulative Time Saved by Hint Type
    sns.barplot(data=hint_stats.sort_values('total_time_saved', ascending=False), x='recommended_hint', y='total_time_saved', palette='viridis', ax=axes[1])
    axes[1].set_title('Cumulative Time Saved by Applied Hint', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Recommended Hint', fontsize=12)
    axes[1].set_ylabel('Total Time Saved (s)', fontsize=12)
    for index, row in hint_stats.sort_values('total_time_saved', ascending=False).reset_index(drop=True).iterrows():
        axes[1].text(index, row['total_time_saved'] + 10, f"{row['total_time_saved']:.1f}s", color='black', ha="center")

    plt.tight_layout()
    plt_path = os.path.join(FIGURES_DIR, 'sata_deep_analysis.png')
    plt.savefig(plt_path, dpi=150)
    plt.close()
    print(f"\n✅ Analytical Figure saved to: {plt_path}")

if __name__ == '__main__':
    main()
