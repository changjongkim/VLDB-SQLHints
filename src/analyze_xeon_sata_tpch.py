
import pandas as pd
import re
import os
import json

def parse_progress_file(filepath):
    queries = []
    if not os.path.exists(filepath):
        return pd.DataFrame()
    with open(filepath, 'r') as f:
        for line in f:
            # Match "[X/Y] q1.sql or tpch_q1.sql - ✓ SUCCESS - 2.022421640s"
            match = re.search(r'\] (?:tpch_)?(q\d+)\.sql - .*? SUCCESS - ([\d.]+)s', line)
            if match:
                qid = match.group(1)
                # Normalize qid to tpch_qX
                if not qid.startswith('tpch_'):
                    qid = 'tpch_' + qid
                time_s = float(match.group(2))
                queries.append({'query_id': qid, 'time': time_s})
    return pd.DataFrame(queries)

# Load data
df_original = parse_progress_file('/root/halo/xeon_results/TPCH/SATA/original/progress.txt')
df_halo = parse_progress_file('/root/halo/xeon_results/TPCH/SATA/halo/progress.txt')

if df_original.empty or df_halo.empty:
    print("Error: Missing data in progress files.")
else:
    # Merge
    df_comp = pd.merge(df_original, df_halo, on='query_id', suffixes=('_orig', '_halo'))
    df_comp['speedup'] = df_comp['time_orig'] / df_comp['time_halo']
    df_comp['time_saved'] = df_comp['time_orig'] - df_comp['time_halo']

    # Load recommendations to see chosen hints
    with open('/root/halo/results/hinted_queries_xeon_v4/recommendation_summary.json') as f:
        recs = json.load(f)
    df_recs = pd.DataFrame([r for r in recs if r['scenario'] == 'Xeon_SATA' and r['benchmark'] == 'TPCH'])
    df_final = pd.merge(df_comp, df_recs[['query_id', 'recommended_hint', 'risk_level', 'reason']], on='query_id', how='left')

    # Summary Statistics
    total_orig = df_comp['time_orig'].sum()
    total_halo = df_comp['time_halo'].sum()
    overall_speedup = total_orig / total_halo

    print(f"### Xeon TPCH SATA Result Analysis ###")
    print(f"Total Queries Compared: {len(df_comp)}")
    print(f"Total Execution Time (Original): {total_orig:.2f}s")
    print(f"Total Execution Time (HALO-P):   {total_halo:.2f}s")
    print(f"Overall Speedup:                 {overall_speedup:.2f}x")
    print(f"Total Time Saved:                {total_orig - total_halo:.2f}s ({(total_orig - total_halo)/60:.1f} mins)")

    print("\nDetailed breakdown:")
    cols = ['query_id', 'time_orig', 'time_halo', 'speedup', 'recommended_hint', 'risk_level']
    print(df_final[cols].sort_values('speedup', ascending=False).to_string(index=False))

    # Tiers
    df_final['tier'] = pd.cut(df_final['time_orig'], bins=[0, 10, 100, 500, 5000], labels=['Fast', 'Med', 'Heavy', 'Mega'])
    tier_summary = df_final.groupby('tier', observed=False).agg({'query_id':'count', 'speedup':'mean', 'time_saved':'sum'}).rename(columns={'query_id':'count'})
    print("\nImpact by Workload Tier:")
    print(tier_summary)
