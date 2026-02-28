from halo_framework import get_halo_v4
from hw_registry import HW_REGISTRY
import logging
import pandas as pd

logging.basicConfig(level=logging.ERROR)

def simulate_buffer_pool(bp_gb):
    # 1. Update Registry
    HW_REGISTRY['Xeon_NVMe']['buffer_pool_gb'] = bp_gb
    
    # 2. Get HALO
    halo = get_halo_v4() # This will re-ingest/re-enrich based on current HW_REGISTRY
    
    # 3. Test significant JOB queries
    job_queries = ['10a', '11a', '12a', '15b', '20a']
    results = []
    
    for qid in job_queries:
        rec = halo.recommend(qid, 'A_NVMe', 'Xeon_NVMe', policy='robust')
        results.append({
            'query_id': qid,
            'hint': rec.recommended_hint or 'NATIVE',
            'reason': rec.reason[:60] + '...'
        })
    return pd.DataFrame(results)

print("--- Simulation: Buffer Pool = 16.0 GB (Baseline) ---")
print(simulate_buffer_pool(16.0))

print("\n--- Simulation: Buffer Pool = 3.0 GB (Low Memory Constraint) ---")
# Reset halo instance for clean state
import halo_framework
halo_framework._halo_v4_instance = None 
print(simulate_buffer_pool(3.0))
