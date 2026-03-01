
import pandas as pd
import re
import os
import json
import numpy as np

def parse_progress_file(filepath):
    queries = []
    if not os.path.exists(filepath):
        return pd.DataFrame()
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\] (?:tpch_)?(q\d+)\.sql - .*? SUCCESS - ([\d.]+)s', line)
            if match:
                qid = match.group(1)
                if not qid.startswith('tpch_'):
                    qid = 'tpch_' + qid
                time_s = float(match.group(2))
                queries.append({'query_id': qid, 'time': time_s})
    return pd.DataFrame(queries)

# 1. Data Loading & Merging
df_origin = parse_progress_file('/root/halo/xeon_results/TPCH/SATA/original/progress.txt')
df_halo = parse_progress_file('/root/halo/xeon_results/TPCH/SATA/halo/progress.txt')

with open('/root/halo/results/hinted_queries_xeon_v4/recommendation_summary.json') as f:
    recs = json.load(f)
df_recs = pd.DataFrame([r for r in recs if r['scenario'] == 'Xeon_SATA' and r['benchmark'] == 'TPCH'])

df = pd.merge(df_origin, df_halo, on='query_id', suffixes=('_orig', '_halo'))
df = pd.merge(df, df_recs[['query_id', 'recommended_hint', 'risk_level', 'reason']], on='query_id', how='left')
df['speedup'] = df['time_orig'] / df['time_halo']
df['time_saved'] = df['time_orig'] - df['time_halo']

# 2. Tier Categorization (Granular)
def get_granular_tier(t):
    if t < 30: return 'P1: Instant (<30s)'
    if t < 100: return 'P2: Fast (30s-100s)'
    if t < 300: return 'P3: Medium (100s-300s)'
    if t < 600: return 'P4: Heavy (300s-600s)'
    return 'P5: Mega (>600s)'

df['tier'] = df['time_orig'].apply(get_granular_tier)

# 3. Full Query List Report
print("="*100)
print(f"{'Query ID':<10} | {'Orig (s)':>10} | {'HALO (s)':>10} | {'Saved (s)':>10} | {'Speedup':>8} | {'Hint':<10} | {'Risk':<8}")
print("-"*100)
for _, row in df.sort_values('time_orig', ascending=False).iterrows():
    hint = str(row['recommended_hint']) if pd.notna(row['recommended_hint']) else 'NATIVE'
    print(f"{row['query_id']:<10} | {row['time_orig']:>10.2f} | {row['time_halo']:>10.2f} | {row['time_saved']:>10.2f} | {row['speedup']:>8.2f}x | {hint:<10} | {row['risk_level']:<8}")

# 4. Tier Aggregation
tier_agg = df.groupby('tier', observed=False).agg(
    count=('query_id', 'count'),
    orig_sum=('time_orig', 'sum'),
    halo_sum=('time_halo', 'sum'),
    saved_sum=('time_saved', 'sum'),
    avg_speedup=('speedup', 'mean')
)
tier_agg['tier_speedup'] = tier_agg['orig_sum'] / tier_agg['halo_sum']

print("\n" + "="*80)
print("GRANULAR WORKLOAD TIER ANALYSIS")
print("="*80)
print(tier_agg.to_string())

# 5. Risk-Benefit Analysis (Why HALO-P worked?)
risk_impact = df.groupby('risk_level', observed=False).agg(
    count=('query_id', 'count'),
    total_saved=('time_saved', 'sum'),
    max_speedup=('speedup', 'max')
)
print("\n" + "="*80)
print("RISK-BENEFIT DISTRIBUTION (Statistically Proof)")
print("="*80)
print(risk_impact.to_string())

# 6. Overall Performance
total_orig = df['time_orig'].sum()
total_halo = df['time_halo'].sum()
print("\n" + "="*80)
print(f"FINAL WORKLOAD SUMMARY")
print(f"Total Original Execution Time : {total_orig:.2f}s")
print(f"Total HALO-P Execution Time   : {total_halo:.2f}s")
print(f"Net Time Saved                : {total_orig - total_halo:.2f}s ({(total_orig - total_halo)/60:.2f} mins)")
print(f"Workload Level Throughput Gain: {(total_orig / total_halo - 1)*100:.2f}% improvement")
print("="*80)
