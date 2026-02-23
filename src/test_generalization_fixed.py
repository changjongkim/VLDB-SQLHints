# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from halo_framework import HaloFramework

def test_generalization_fixed():
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')
    
    halo = HaloFramework()
    halo.train(df_ops, df_q)
    halo._build_query_vectors(df_ops)
    
    # IMPROVEMENT: Use ALL known successful hints across ALL environments 
    # as a "Knowledge Base" for the new query.
    
    envs = ['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA']
    
    print("=== FIXED Generalization Test (Using Full Knowledge Base) ===")
    
    for env in envs:
        queries = df_q[df_q['env_id'] == env]['query_id'].unique()
        total = 0
        recommended = 0
        
        for qid in queries:
            # 1. New Query arrives in 'env'
            plan_df = df_ops[(df_ops['env_id'] == env) & (df_ops['query_id'] == qid) & (df_ops['hint_set'] == 'baseline')]
            if plan_df.empty: continue
            total += 1
            
            # 2. Find similar queries ANYWHERE in the knowledge base
            # We look for queries that had a successful hint in ANY environment
            similar = halo.find_similar_queries(env, plan_df, top_n=5)
            
            found_safe_hint = False
            for sim_qid, dist in similar:
                # 3. Does this similar query have a hint that works WELL in any env?
                # We check B_NVMe or B_SATA since they have the best hint candidates
                for src_env_for_hint in ['B_NVMe', 'B_SATA', 'A_NVMe']:
                    res = halo.recommend(sim_qid, src_env_for_hint, env, policy='robust')
                    if res.recommended_hint:
                        recommended += 1
                        found_safe_hint = True
                        break
                if found_safe_hint: break
        
        rate = (recommended / total * 100) if total > 0 else 0
        print(f"Environment: {env:<8} | Total New Qs: {total:<4} | Recommended: {recommended:<4} | Rate: {rate:>5.1f}%")

if __name__ == "__main__":
    test_generalization_fixed()
