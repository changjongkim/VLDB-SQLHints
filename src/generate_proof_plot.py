# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_logic_proof_plot():
    # We want a plot that shows:
    # X-axis: Predicted Risk (Delta)
    # Y-axis: Source Speedup (Potential Gain)
    # Points colored by whether they were recommended or rejected.
    
    audit_path = '/root/halo/results/halo_full_audit.csv'
    if not os.path.exists(audit_path): return
    
    df = pd.read_csv(audit_path)
    # Focus on B_NVMe -> A_NVMe for the CPU story
    df = df[df['Xfer'] == 'B_NVMe->A_NVMe'].copy()
    
    # Extract max delta from High_Risk_Ops
    def get_max_delta(ops_str):
        if pd.isna(ops_str) or ops_str == "": return 0.0
        import re
        deltas = re.findall(r'\+([0-9.]+)', ops_str)
        if deltas:
            return max([float(d) for d in deltas])
        return 0.0

    df['Max_Risk'] = df['High_Risk_Ops'].apply(get_max_delta)
    df['Decision_Type'] = df['Decision'].apply(lambda x: 'NATIVE' if x == 'NATIVE' else 'RECOMMEND')
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='Max_Risk', y='Source_Spd', hue='Decision_Type', 
                    palette={'NATIVE': 'red', 'RECOMMEND': 'green'}, alpha=0.7, s=100)
    
    plt.axvline(0.5, color='orange', linestyle='--', label='Risk Threshold (0.5)')
    plt.yscale('log')
    plt.title('HALO-R Decision Logic: B_NVMe (Server) -> A_NVMe (Desktop)\nWhy we stay Native despite High Source Speedup', fontsize=14)
    plt.xlabel('Predicted Operator Slowdown (Max $\sigma$)', fontsize=12)
    plt.ylabel('Source Environment Speedup (Log Scale)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Annotate some clear cases
    top_saved = df[df['Source_Spd'] > 10].sort_values('Max_Risk', ascending=False).head(3)
    for _, row in top_saved.iterrows():
        plt.annotate(f"Query {row['Query']}\n({row['Source_Spd']:.0f}x src)", 
                     (row['Max_Risk'], row['Source_Spd']),
                     textcoords="offset points", xytext=(10,0), ha='left', fontsize=9, color='darkred')

    output_path = '/root/halo/results/figures/proof_of_logic_cpu.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved proof of logic plot to {output_path}")

if __name__ == "__main__":
    generate_logic_proof_plot()
