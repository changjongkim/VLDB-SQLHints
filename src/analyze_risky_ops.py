# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_regressions_avoided():
    # Load audit results
    audit_path = '/root/halo/results/halo_full_audit.csv'
    if not os.path.exists(audit_path):
        print("Audit file not found.")
        return
    
    df = pd.read_csv(audit_path)
    
    # Cases where HALO-R stayed Native but Source Speedup was > 1.0
    # These are the interesting "Decisions"
    fallbacks = df[(df['Decision'] == 'NATIVE') & (df['Source_Spd'] > 1.1)].copy()
    
    # Cross-reference with Ground Truth to see if it was a "good" fallback or "missed opportunity"
    # Status is already in the audit: 'Correct' if actual_spd >= 0.95 else 'Regression'
    # Actually, if we chose NATIVE, actual_spd is 1.0 by definition.
    # We need to know what the ACTUAL speedup would have been if we picked the hint.
    
    # Filter for the most interesting scenario: A_NVMe -> B_SATA or B_NVMe -> A_NVMe
    
    # Let's visualize the "Risk Score" vs "Actual Performance" for some queries
    print("Top Fallbacks Analyzed:")
    print(fallbacks[['Xfer', 'Query', 'Source_Spd', 'High_Risk_Ops']].head(20))
    
    # Create a plot showing Source Speedup vs High Risk Ops count
    # for all recommendations
    plt.figure(figsize=(10, 6))
    
    # We need a dataframe of ALL candidates for this.
    # For now, let's use the audit df but focus on the 'Source_Spd' and 'Reason'
    
    # Let's create a bar chart of "Top Operators causing Native Fallbacks"
    all_risky_ops = []
    for ops in fallbacks['High_Risk_Ops'].dropna():
        # format: "OP(+0.XX), OP(+0.XX)"
        parts = ops.split(', ')
        for p in parts:
            op_name = p.split('(')[0]
            all_risky_ops.append(op_name)
    
    risky_counts = pd.Series(all_risky_ops).value_counts()
    
    plt.figure(figsize=(10, 6))
    risky_counts.plot(kind='bar', color='salmon')
    plt.title('Operators Most Frequently Triggering Safety Fallbacks\n(Predicted SLOWDOWN during HW Transfer)', fontsize=12)
    plt.ylabel('Number of Fallbacks Triggered', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('/root/halo/results/figures/risky_operators_summary.png', dpi=300)
    print("\nSaved risky operators plot.")

if __name__ == "__main__":
    analyze_regressions_avoided()
