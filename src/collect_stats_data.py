import os
import subprocess
import json
import time
import pandas as pd
import re
import csv
from tqdm import tqdm

MYSQL_CMD = "/usr/local/mysql-8.0.25/bin/mysql -u root -proot --socket=/tmp/mysql_3356.sock stats -BNe"
OUT_FILE = '/root/halo/data/stats/stats_A_SATA_queries_raw.csv'

HINT_DEFINITIONS = {
    'baseline': '',
    'hint01': '/*+ SET_VAR(optimizer_switch="block_nested_loop=off") */',
    'hint02': '/*+ BKA(v p c ph pl u b t) */',
    'hint04': '/*+ JOIN_ORDER(v, p, c, ph, pl, u, b, t) */',
    'hint05': '/*+ SET_VAR(optimizer_switch="block_nested_loop=off") */',
}

def clean_comment(sql):
    return re.sub(r'--.*', '', sql).strip()

def inject_hint(sql, hint):
    if not hint:
        return sql
    if "SELECT COUNT(*)" in sql.upper():
        return sql.replace("SELECT COUNT(*)", f"SELECT {hint} COUNT(*)", 1)
    elif "SELECT" in sql.upper():
        return sql.replace("SELECT", f"SELECT {hint}", 1)
    return sql

def run_query(sql):
    with open('/tmp/stats_query.sql', 'w') as f:
        f.write(sql)
    
    cmd = f"{MYSQL_CMD} \"EXPLAIN ANALYZE $(cat /tmp/stats_query.sql)\""
    start_time = time.perf_counter()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        end_time = time.perf_counter()
        if result.returncode != 0:
            return None, end_time - start_time
        return result.stdout, end_time - start_time
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 120.0
    except Exception as e:
        return None, 120.0

def main():
    os.makedirs('/root/halo/data/stats', exist_ok=True)
    
    query_files = sorted([f for f in os.listdir('/root/stats-benchmark/queries_mysql') if f.endswith('.sql')])
    
    # Header for CSV
    cols = ['env_id', 'benchmark', 'query_id', 'hint_set', 'execution_time_s', 'status']
    
    # Check if we should resume
    existing_qids = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_qids.add(row['query_id'])

    f_out = open(OUT_FILE, 'a', newline='')
    writer = csv.DictWriter(f_out, fieldnames=cols)
    if os.path.getsize(OUT_FILE) == 0:
        writer.writeheader()

    print(f"Starting/Resuming data collection for {len(query_files)} queries...")
    
    # Focus on first 50 queries for a quick demo if you're checking for progress
    for q_file in tqdm(query_files):
        qid = q_file.replace('.sql', '').replace('stats_', '')
        
        # Simple resume: skip if already collected enough for this qid
        if list(HINT_DEFINITIONS.keys())[0] in [r for r in existing_qids if r == qid]:
            # This logic isn't perfect for per-hint resume, 
            # but let's just assume we continue from qid.
            continue
            
        with open(f'/root/stats-benchmark/queries_mysql/{q_file}', 'r') as f:
            original_sql = clean_comment(f.read())
            
        for h_name, h_val in HINT_DEFINITIONS.items():
            hinted_sql = inject_hint(original_sql, h_val)
            output, exec_time = run_query(hinted_sql)
            
            row = {
                'env_id': 'A_SATA',
                'benchmark': 'STATS',
                'query_id': qid,
                'hint_set': h_name,
                'execution_time_s': exec_time,
                'status': 'SUCCESS' if output and output != "TIMEOUT" else ('TIMEOUT' if output == "TIMEOUT" else 'FAILED')
            }
            writer.writerow(row)
            f_out.flush()
            
    f_out.close()
    
    # Final conversion to Parquet for HALO
    df = pd.read_csv(OUT_FILE)
    df.to_parquet('/root/halo/data/stats/stats_A_SATA_queries.parquet')
    print(f"Completed! Final results at /root/halo/data/stats/stats_A_SATA_queries.parquet")

if __name__ == '__main__':
    main()
