import pandas as pd
import json

df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')

# Dictionary to store baselines
baselines = {}
for _, row in df_q[df_q['hint_set'] == 'baseline'].iterrows():
    baselines[(row['env_id'], row['benchmark'], row['query_id'])] = row['execution_time_s']

# Let's look at all valid cross-environment pairs.
pairs = [
    ('A_NVMe', 'A_SATA'), ('A_NVMe', 'B_NVMe'), ('A_NVMe', 'B_SATA'),
    ('A_SATA', 'A_NVMe'), ('A_SATA', 'B_NVMe'), ('A_SATA', 'B_SATA'),
    ('B_NVMe', 'A_NVMe'), ('B_NVMe', 'A_SATA'), ('B_NVMe', 'B_SATA'),
    ('B_SATA', 'A_NVMe'), ('B_SATA', 'A_SATA'), ('B_SATA', 'B_NVMe'),
]

# Note: for JOB, we only have actual hint performance on A_NVMe, A_SATA (Baseline only), B_NVMe(Baseline only), B_SATA(Baseline and hints 01,02,04,05)
# Wait, for JOB unified_queries.parquet? Let's check what's inside.

records = []
for _, row in df_q[df_q['hint_set'] != 'baseline'].iterrows():
    tgt_env = row['env_id']
    bench = row['benchmark']
    qid = row['query_id']
    hint = row['hint_set']
    tgt_time = row['execution_time_s']
    
    tgt_base = baselines.get((tgt_env, bench, qid))
    if tgt_base is None: continue
    
    actual_tgt_speedup = tgt_base / max(tgt_time, 0.001)
    if actual_tgt_speedup <= 1.05: continue # only care about accelerations
    
    # We pretend this hint was recommended for ALL source environments that we transfer from.
    # In reality, HALO only recommends it if src_speedup > 1.0 and safe. Let's just mock the typical transfer logic
    # by finding the Max Source Baseline vs Source Hint speedup.
    for src_env in ['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA']:
        if src_env == tgt_env: continue
        
        src_base = baselines.get((src_env, bench, qid))
        if src_base is None: continue
        
        src_hint_times = df_q[(df_q['env_id']==src_env) & (df_q['benchmark']==bench) & (df_q['query_id']==qid) & (df_q['hint_set']==hint)]
        if src_hint_times.empty: continue
        src_time = src_hint_times['execution_time_s'].min()
        src_speedup = src_base / max(src_time, 0.001)
        
        if src_speedup > 1.0: # Means the hint was "good" on source, so HALO would consider it
            records.append({
                'Transfer': f"{src_env} → {tgt_env}",
                'Benchmark': bench,
                'Query': qid,
                'Hint': hint,
                'Source Speedup (Predicted by HALO)': src_speedup,
                'Actual Target Speedup': actual_tgt_speedup
            })

df = pd.DataFrame(records)
if not df.empty:
    tpch = df[df['Benchmark'] == 'TPCH'].sort_values('Actual Target Speedup', ascending=False)
    # Deduplicate by Actual Target Speedup + Query
    tpch = tpch.drop_duplicates(subset=['Query', 'Hint', 'Actual Target Speedup'])
    job = df[df['Benchmark'] == 'JOB'].sort_values('Actual Target Speedup', ascending=False)
    job = job.drop_duplicates(subset=['Query', 'Hint', 'Actual Target Speedup'])

    print("--- TPCH TOP ---")
    print(tpch.head(10).to_string())
    print()
    print("--- JOB TOP ---")
    print(job.head(10).to_string())
else:
    print("No valid transfers found.")
