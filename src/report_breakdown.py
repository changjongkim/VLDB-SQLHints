# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np

def generate_report():
    results_path = '/root/halo/results/halo_framework_results.json'
    if not os.path.exists(results_path):
        print("Results file not found. Run halo_framework.py first.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    summary = []
    for t in data['transfers']:
        src, tgt = t['transfer'].split(' → ')
        s_parts = src.split('_')
        t_parts = tgt.split('_')
        
        storage_diff = s_parts[1] != t_parts[1]
        cpu_diff = s_parts[0] != t_parts[0]
        
        if src == tgt: cat = 'Home'
        elif storage_diff and cpu_diff: cat = 'Both (Storage+CPU)'
        elif storage_diff: cat = 'Storage-only'
        elif cpu_diff: cat = 'CPU-only'
        else: cat = 'Home'
        
        summary.append({
            'Category': cat,
            'G_Reg': t['halo_g']['n_regressions'],
            'R_Reg': t['halo_r']['n_regressions'],
            'Rescues': len(t['rescues']),
            'G_Spd': t['halo_g']['avg_speedup'],
            'R_Spd': t['halo_r']['avg_speedup']
        })
        
    df = pd.DataFrame(summary)
    grouped = df.groupby('Category').agg({
        'G_Reg': 'sum',
        'R_Reg': 'sum',
        'Rescues': 'sum',
        'G_Spd': 'mean',
        'R_Spd': 'mean'
    })
    
    grouped['Reg_Reduction'] = (1 - grouped['R_Reg'] / grouped['G_Reg'].replace(0, 1)) * 100
    grouped['Reg_Reduction'] = grouped['Reg_Reduction'].where(grouped['G_Reg'] > 0, 0)
    
    print("\n=== HALO Environment-Type Breakdown Report ===")
    print(grouped.to_string())
    
    # Also get high-risk classification metrics from sigma results
    sigma_path = '/root/halo/results/sigma_results_v2.json'
    if os.path.exists(sigma_path):
        with open(sigma_path, 'r') as f:
            sd = json.load(f)
        print("\n=== σ Model Classification Metrics (|σ| > 0.5) ===")
        m = sd.get('high_risk_metrics', {})
        print(f"  Precision: {m.get('precision', 0):.3f}")
        print(f"  Recall:    {m.get('recall', 0):.3f}")
        print(f"  F1-Score:  {m.get('f1', 0):.3f}")
        print(f"  Overall Direction Accuracy: {sd.get('direction_accuracy', 0):.1f}%")

if __name__ == "__main__":
    import os
    generate_report()
