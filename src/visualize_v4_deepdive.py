
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from halo_framework import get_halo_v4
from hw_registry import HW_REGISTRY
from sigma_model_v4 import engineer_features_v4

# Output directory
FIGURES_DIR = '/root/halo/results/v4/figures/deep_dive'
os.makedirs(FIGURES_DIR, exist_ok=True)

def sensitivity_analysis():
    print("Running Hardware Sensitivity Analysis...")
    halo = get_halo_v4()
    qid = 'tpch_q9'
    hs = 'hint05'
    source = 'A_NVMe'
    target = 'Xeon_NVMe'
    
    # Get a representative operator (e.g., a large Join)
    op = halo._operator_index[(source, qid, hs)][0]
    
    # Sweep CPU Clock Ratio
    clocks = np.linspace(1.0, 5.0, 20)
    mu_list = []
    sigma_list = []
    
    base_pair = halo._get_env_pair_metadata(source, target)
    
    for c in clocks:
        # Simulate hardware change
        temp_pair = base_pair.copy()
        temp_pair['cpu_clock_ratio'] = np.log(c / 5.2) # Normalized to i9
        
        feat_dict = engineer_features_v4(op, op, temp_pair, halo.risk_dict)
        X = np.array([[feat_dict[c] for c in halo.feature_cols]], dtype=np.float32)
        X_tensor = torch.FloatTensor(halo.scaler.transform(X)).to(halo.device)
        
        with torch.no_grad():
            mu, log_sigma = halo.sigma_model(X_tensor)
            mu_list.append(mu.item())
            sigma_list.append(torch.exp(log_sigma).item())

    # Plot Sensitivity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(clocks, mu_list, 'b-o', label='Predicted Mean (mu)')
    ax1.set_xlabel('Simulated CPU Clock (GHz)', fontsize=12)
    ax1.set_ylabel('Predicted Speedup (log-space mu)', color='b', fontsize=12)
    
    ax2 = ax1.twinx()
    ax2.plot(clocks, sigma_list, 'r-s', label='Uncertainty (sigma)')
    ax2.set_ylabel('Model Uncertainty (sigma)', color='r', fontsize=12)
    
    plt.title(f'Hardware Sensitivity Analysis: CPU Clock vs Prediction ({qid})', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(FIGURES_DIR, 'sensitivity_cpu.png'), dpi=150)
    plt.close()
    print(f"Sensitivity Analysis saved to {FIGURES_DIR}")

def operator_risk_heatmap():
    """Deep Dive: Visualize risk across the operator tree for a complex query."""
    print("Generating Operator Risk Heatmap...")
    halo = get_halo_v4()
    qid = '10a'
    hs = 'hint02'
    source = 'A_NVMe'
    target = 'Xeon_NVMe'
    
    rec = halo.recommend(qid, source, target, policy='performance')
    if not rec.details:
        print(f"No details found for {qid}")
        return
        
    # Find the candidate that matches the recommended hint
    cand = next((c for c in rec.details if c.hint_set == rec.recommended_hint), rec.details[0])
    ops = cand.operator_details 
    
    df_ops = pd.DataFrame([
        {'Type': o.op_type, 'Table': o.table_alias, 'Depth': o.depth, 'Risk': o.upper_bound} 
        for o in ops
    ])
    
    plt.figure(figsize=(12, 6))
    sns_plot = sns.barplot(data=df_ops, x='Type', y='Risk', hue='Table', palette='viridis')
    plt.title(f'Structural Risk Distribution: {qid} Plan Tree', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.axhline(1.2, color='red', linestyle='--', label='Safety Threshold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'structural_risk_distribution.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    import seaborn as sns
    sensitivity_analysis()
    operator_risk_heatmap()
