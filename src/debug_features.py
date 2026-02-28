from halo_framework import get_halo_v4
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)

halo = get_halo_v4()
qid = 'tpch_q9'
source_env = 'A_NVMe'
target_env = 'Xeon_NVMe'

# Re-assess risk for one hint candidate to see features
hs = 'hint05'
pair = halo._get_env_pair_metadata(source_env, target_env)
ops = halo._operator_index.get((source_env, qid, hs), [])

if ops:
    op = ops[0]
    from sigma_model_v4 import engineer_features_v4
    feat_dict = engineer_features_v4(op, op, pair, halo.risk_dict)
    
    print("\n=== Feature Inspection for Xeon NVMe ===")
    important_feats = [
        'cpu_core_ratio', 'cpu_clock_ratio', 'iops_ratio', 
        'dot_product_alignment', 'demand_supply_mismatch',
        'log_actual_rows'
    ]
    for f in important_feats:
        print(f"  {f}: {feat_dict.get(f, 'MISSING')}")
    
    # Check model prediction
    X = np.array([[feat_dict.get(c, 0.0) for c in halo.feature_cols]], dtype=np.float32)
    X_scaled = halo.scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(halo.device)
    
    mu, sigma = halo.sigma_model(X_tensor)
    print(f"\nModel Raw Prediction: mu={mu.item():.3f}, sigma={torch.exp(sigma).item():.3f}")

    # Check EPYC comparison
    pair_epyc = halo._get_env_pair_metadata(source_env, 'B_NVMe')
    feat_epyc = engineer_features_v4(op, op, pair_epyc, halo.risk_dict)
    print("\n=== Feature Comparison (A -> EPYC) ===")
    for f in important_feats:
         print(f"  {f}: {feat_epyc.get(f, 'MISSING')}")
    X_e = np.array([[feat_epyc.get(c, 0.0) for c in halo.feature_cols]], dtype=np.float32)
    X_e_scaled = halo.scaler.transform(X_e)
    X_e_tensor = torch.FloatTensor(X_e_scaled).to(halo.device)
    mu_e, _ = halo.sigma_model(X_e_tensor)
    print(f"\nEPYC Raw Prediction: mu={mu_e.item():.3f}")
