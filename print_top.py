import pandas as pd

df = pd.read_parquet('/root/halo/data/unified_queries.parquet')

# Need to calculate speedup over native (baseline) for each query in its environment.
baselines = df[df['hint_set'] == 'baseline'][['env_id', 'benchmark', 'query_id', 'execution_time_s']].rename(columns={'execution_time_s': 'base_time'})

# Get max speedup for each query across all hints
hints = df[df['hint_set'] != 'baseline'].groupby(['env_id', 'benchmark', 'query_id', 'hint_set'])['execution_time_s'].min().reset_index()
merged = pd.merge(hints, baselines, on=['env_id', 'benchmark', 'query_id'])
merged['speedup'] = merged['base_time'] / merged['execution_time_s']

# Group by to find best speedup
best = merged.loc[merged.groupby(['env_id', 'benchmark', 'query_id'])['speedup'].idxmax()]

# Get top 10 unique queries per benchmark
tpch_top = best[best['benchmark'] == 'TPCH'].sort_values('speedup', ascending=False).drop_duplicates('query_id').head(10)
job_top = best[best['benchmark'] == 'JOB'].sort_values('speedup', ascending=False).drop_duplicates('query_id').head(10)

print('### Top 10 Performance Accelerations (TPC-H)')
print('| Rank | Query | Environment | Best Hint | Speedup |')
print('|:---:|:---:|:---:|:---:|:---:|')
for i, (_, row) in enumerate(tpch_top.iterrows()):
    print(f'| {i+1} | `{row.query_id}` | {row.env_id} | `{row.hint_set}` | **{row.speedup:.2f}x** |')

print('')
print('### Top 10 Performance Accelerations (JOB)')
print('| Rank | Query | Environment | Best Hint | Speedup |')
print('|:---:|:---:|:---:|:---:|:---:|')
for i, (_, row) in enumerate(job_top.iterrows()):
    print(f'| {i+1} | `{row.query_id}` | {row.env_id} | `{row.hint_set}` | **{row.speedup:.2f}x** |')
