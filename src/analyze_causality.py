
from halo_framework import get_halo_v4
from hw_registry import HW_REGISTRY, compute_hw_features
from sigma_model_v4 import engineer_features_v4
import torch
import numpy as np

def analyze_prediction_causality(qid, hint_set, src, tgt):
    halo = get_halo_v4()
    # 1. Hardware Specs (Input A)
    src_hw = HW_REGISTRY.get(src)
    tgt_hw = HW_REGISTRY.get(tgt)
    pair = compute_hw_features(src, tgt)

    # 2. Performance in Source (Input B)
    base_src = halo._query_baselines.get((src, qid))
    hint_src_row = halo.df_queries[(halo.df_queries['env_id'] == src) & (halo.df_queries['query_id'] == qid) & (halo.df_queries['hint_set'] == hint_set)].sort_values('execution_time_s').iloc[0]
    src_speedup = base_src / hint_src_row['execution_time_s']

    # 3. Model Prediction Logic
    rec = halo.recommend(qid, src, tgt, policy='performance')
    
    # Analyze the most critical join in this query
    # Find operators for this hint
    ops = halo._operator_index.get((src, qid, hint_set), [])
    critical_op = None
    max_rows = 0
    for op in ops:
        if 'JOIN' in op['op_type'] and op['actual_rows'] > max_rows:
            max_rows = op['actual_rows']
            critical_op = op

    if not critical_op: critical_op = ops[0]

    # Feature extraction for critical operator
    feat = engineer_features_v4(critical_op, critical_op, pair, halo.risk_dict)
    
    print(f"\n[QUERY DEEP DIVE: {qid}]")
    print(f"  Target Hardware: {tgt} (Cores: {tgt_hw['cpu_cores']}, Clock: {tgt_hw['cpu_base_ghz']}GHz, Buffer: {tgt_hw['buffer_pool_gb']}GB)")
    print(f"  Source Speedup (i9): {src_speedup:.2f}x")
    
    print("\n[KEY HARDWARE-SOFTWARE ALIGNMENT FEATURES]")
    print(f"  1. Vector Alignment (IOPS x CPU): {feat['dot_product_alignment']:.3f}")
    print(f"  2. Demand/Supply Mismatch: {feat['demand_supply_mismatch']:.3f} (Lower is better)")
    print(f"  3. Fits in Buffer (Xeon): {feat['fits_in_buffer']} (1=Safe, 0=Risk)")
    print(f"  4. CPU Complexity Load: {feat['subtree_total_work']:.2e} (Normalized ops)")
    
    # Model Output
    X = np.array([[feat[c] for c in halo.feature_cols]])
    X_tensor = torch.FloatTensor(halo.scaler.transform(X)).to(halo.device)
    with torch.no_grad():
        mu, log_sigma = halo.sigma_model(X_tensor)
        sigma = torch.exp(log_sigma)
        
    print(f"\n[MODEL PREDICTION FOR CRITICAL OPERATOR]")
    print(f"  Predicted Gain (mu): {mu.item():.3f} (Log scale Delta)")
    print(f"  Uncertainty (sigma): {sigma.item():.3f}")
    print(f"  Conformal Upper Bound Risk: {mu.item() + 2.14 * sigma.item():.3f}")
    
    print(f"\n[FINAL RECOMMENDATION STATUS]")
    print(f"  Result: {rec.recommended_hint or 'NATIVE'} (Reason: {rec.reason})")

if __name__ == "__main__":
    analyze_prediction_causality('tpch_q17', 'hint02', 'A_NVMe', 'Xeon_NVMe')
