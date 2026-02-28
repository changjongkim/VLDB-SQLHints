
import pandas as pd
import re
import os

def parse_progress_file(filepath):
    queries = []
    with open(filepath, 'r') as f:
        for line in f:
            # Match "[X/Y] 1a.sql - ✓ SUCCESS - 2.022421640s"
            match = re.search(r'\] ([\w.]+)\.sql - .*? SUCCESS - ([\d.]+)s', line)
            if match:
                qid = match.group(1)
                time_s = float(match.group(2))
                queries.append({'query_id': qid, 'time': time_s})
    return pd.DataFrame(queries)

# Load data
df_original = parse_progress_file('/root/halo/xeon_results/JOB/SATA/original/progress.txt')
df_halo = parse_progress_file('/root/halo/xeon_results/JOB/SATA/halo/progress.txt')

# Merge
df_comp = pd.merge(df_original, df_halo, on='query_id', suffixes=('_orig', '_halo'))
df_comp['speedup'] = df_comp['time_orig'] / df_comp['time_halo']

# Summary Statistics
total_orig = df_comp['time_orig'].sum()
total_halo = df_comp['time_halo'].sum()
overall_speedup = total_orig / total_halo

print(f"### Xeon JOB SATA Result Analysis ###")
print(f"Total Queries Compared: {len(df_comp)}")
print(f"Total Execution Time (Original): {total_orig:.2f}s")
print(f"Total Execution Time (HALO-P):   {total_halo:.2f}s")
print(f"Overall Gear-Shift Speedup:      {overall_speedup:.2f}x")

# Top Speedups
print("\nTop 10 Speedups:")
print(df_comp.sort_values('speedup', ascending=False).head(10)[['query_id', 'time_orig', 'time_halo', 'speedup']].to_string(index=False))

# Regressions
regressions = df_comp[df_comp['speedup'] < 0.95]
print(f"\nSignificant Regressions (<0.95x): {len(regressions)}")
if not regressions.empty:
    print(regressions.sort_values('speedup').head(5)[['query_id', 'time_orig', 'time_halo', 'speedup']].to_string(index=False))

# Histogram/Summary
bins = [0, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0, 5.0, 100.0]
df_comp['bin'] = pd.cut(df_comp['speedup'], bins=bins)
print("\nSpeedup Distribution:")
print(df_comp['bin'].value_counts().sort_index())
