import pandas as pd
import json

df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')

# Dictionary to store baselines
baselines = {}
for _, row in df_q[df_q['hint_set'] == 'baseline'].iterrows():
    baselines[(row['env_id'], row['benchmark'], row['query_id'])] = row['execution_time_s']

# Let's search all meaningful cross-environment pairs
pairs = [
    ('A_NVMe', 'A_SATA'), ('A_NVMe', 'B_NVMe'), ('A_NVMe', 'B_SATA'),
    ('A_SATA', 'A_NVMe'), ('A_SATA', 'B_NVMe'), ('A_SATA', 'B_SATA'),
    ('B_NVMe', 'A_NVMe'), ('B_NVMe', 'A_SATA'), ('B_NVMe', 'B_SATA'),
    ('B_SATA', 'A_NVMe'), ('B_SATA', 'A_SATA'), ('B_SATA', 'B_NVMe'),
]

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
    if actual_tgt_speedup <= 1.0: continue # only care about accelerations
    
    # Evaluate across all valid sources
    for src_env in ['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA']:
        if src_env == tgt_env: continue
        
        src_base = baselines.get((src_env, bench, qid))
        if src_base is None: continue
        
        # Get source hint time
        # We might have multiple executions, so we take the minimum (best time)
        src_times = df_q[(df_q['env_id']==src_env) & (df_q['benchmark']==bench) & (df_q['query_id']==qid) & (df_q['hint_set']==hint)]
        if src_times.empty: continue
        
        src_time = src_times['execution_time_s'].min()
        src_speedup = src_base / max(src_time, 0.001)
        
        # The logic: If this hint was good on the source env, HALO would consider it.
        # We want to show how it performed when actually transferred to target.
        if src_speedup > 1.05: 
            records.append({
                'Transfer': f"{src_env} → {tgt_env}",
                'Benchmark': bench,
                'Query': qid,
                'Hint': hint,
                'Source_Speedup': src_speedup,
                'Actual_Target_Speedup': actual_tgt_speedup
            })

df = pd.DataFrame(records)
# Only keep the "best" hint transfer scenario per query overall (across all transfers config)
# Or we just find the absolute 10 best transfer achievements per benchmark!
tpch = df[df['Benchmark'] == 'TPCH'].sort_values('Actual_Target_Speedup', ascending=False)
tpch = tpch.drop_duplicates(subset=['Query']).head(10)

job = df[df['Benchmark'] == 'JOB'].sort_values('Actual_Target_Speedup', ascending=False)
job = job.drop_duplicates(subset=['Query']).head(10)

print("### Top 10 Cross-Environment Accelerations (TPC-H)")
print("| Rank | Transfer | Query | Hint | Predicted Trend (Source) | Actual Target Speedup |")
print("|:---:|:---|:---:|:---:|:---:|:---:|")
for i, (_, row) in enumerate(tpch.iterrows()):
    print(f"| {i+1} | {row['Transfer']} | `{row['Query']}` | `{row['Hint']}` | {row['Source_Speedup']:.2f}x | **{row['Actual_Target_Speedup']:.2f}x** |")

print("\n### Top 10 Cross-Environment Accelerations (JOB)")
print("| Rank | Transfer | Query | Hint | Predicted Trend (Source) | Actual Target Speedup |")
print("|:---:|:---|:---:|:---:|:---:|:---:|")
for i, (_, row) in enumerate(job.iterrows()):
    print(f"| {i+1} | {row['Transfer']} | `{row['Query']}` | `{row['Hint']}` | {row['Source_Speedup']:.2f}x | **{row['Actual_Target_Speedup']:.2f}x** |")

