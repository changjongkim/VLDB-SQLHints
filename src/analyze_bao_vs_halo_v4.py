import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configs
PREDICTIONS_PATH = '/root/halo/results/v4/sigma_predictions_v4.parquet'
OUT_DIR = '/root/halo/assets'
os.makedirs(OUT_DIR, exist_ok=True)

# Colors
COLORS = {'Bao': '#EA4335', 'HALO': '#34A853', 'Native': '#9AA0A6'}

def run_bao_vs_halo_experiment():
    print("Loading v4 predictions to simulate policy choices...")
    df = pd.read_parquet(PREDICTIONS_PATH)
    
    # We simulate policy making at the Hint Set level for each Query in a Target Environment
    # Group operator predictions to get "Query-Hint" level expectations
    # A regression at ANY operator can ruin a query, so we use max() for risk/time delta
    
    qh_df = df.groupby(['pair_label', 'query_id', 'hint_set']).agg(
        actual_delta_max=('delta_time', 'max'),       # Actual max regression
        pred_mu_sum=('predicted_mu', 'sum'),          # Total expected predicted time
        pred_sigma_mean=('predicted_sigma', 'mean'),  # Average uncertainty
        upper_bound_max=('upper_bound', 'max')        # Max safety bound (Halo's metric)
    ).reset_index()

    # Define Policies for each (pair_label, query_id)
    # 1. NATIVE Baseline: delta_time = 0
    # 2. BAO-LIKE: Pick the hint that minimizes "pred_mu_sum" (Most greedy!)
    # 3. HALO: Pick hint with best "pred_mu_sum" ONLY IF "upper_bound_max" <= 0.1 (Safety Gate)

    results = []
    
    grouped = qh_df.groupby(['pair_label', 'query_id'])
    for (pair, qid), group in grouped:
        # What is the actual regression if we do nothing? (0.0)
        
        # BAO-LIKE Choice
        bao_choice = group.loc[group['pred_mu_sum'].idxmin()]
        bao_actual_delta = bao_choice['actual_delta_max']
        
        # HALO Choice
        safe_hints = group[group['upper_bound_max'] <= 0.1]
        if len(safe_hints) > 0:
            halo_choice = safe_hints.loc[safe_hints['pred_mu_sum'].idxmin()]
            halo_actual_delta = halo_choice['actual_delta_max']
            halo_status = 'RECOMMEND'
        else:
            halo_actual_delta = 0.0 # Native Fallback!
            halo_status = 'NATIVE'
            
        results.append({
            'pair': pair,
            'query_id': qid,
            'bao_delta': bao_actual_delta,
            'halo_delta': halo_actual_delta,
            'halo_status': halo_status
        })
    
    res_df = pd.DataFrame(results)
    
    # Metrics
    # >0 implies REGRESSION (slower than baseline)
    # <0 implies SPEEDUP (faster than baseline)
    bailout_rate = (res_df['halo_status'] == 'NATIVE').mean() * 100
    
    bao_regressions = (res_df['bao_delta'] > 0.1).sum()
    halo_regressions = (res_df['halo_delta'] > 0.1).sum()
    
    bao_severe = (res_df['bao_delta'] > 0.5).sum()
    halo_severe = (res_df['halo_delta'] > 0.5).sum()
    
    print("-" * 60)
    print("🌟 BAO-like vs HALO v4 Policy Showdown (Zero-Shot Hardware)")
    print("-" * 60)
    print(f"Total Queries Tested: {len(res_df):,}")
    print(f"HALO Fallback (NATIVE) Rate: {bailout_rate:.1f}%")
    print(f"BAO Regressions (>10% slow): {bao_regressions:,} queries")
    print(f"HALO Regressions (>10% slow): {halo_regressions:,} queries ({-((bao_regressions-halo_regressions)/bao_regressions)*100:.1f}%)")
    print(f"BAO Severe Drops (>50% slow): {bao_severe:,} queries!")
    print(f"HALO Severe Drops (>50% slow): {halo_severe:,} queries")

    # Generate Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("HALO vs Bao-like Baseline (Zero-Shot Target Hardware)", fontsize=18, fontweight='bold')
    
    # (a) Regression Counts (The Danger of Greed)
    ax = axes[0]
    methods = ['Bao-like (Greedy)', 'HALO v4 (Safety-Gated)']
    minor_regs = [bao_regressions - bao_severe, halo_regressions - halo_severe]
    severe_regs = [bao_severe, halo_severe]
    
    idx = np.arange(2)
    width = 0.5
    ax.bar(idx, minor_regs, width, label='Minor Regression (>10%)', color='#fb8c00')
    ax.bar(idx, severe_regs, width, bottom=minor_regs, label='Severe Regression (>50%)', color='#d32f2f')
    
    for i, (m, s) in enumerate(zip(minor_regs, severe_regs)):
        ax.text(i, m + s + 20, f"Total: {m+s:,}", ha='center', fontsize=12, fontweight='bold')
        if s > 0:
            ax.text(i, m + s/2, f"{s:,}", ha='center', va='center', color='white', fontweight='bold')
            
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Regressed Queries")
    ax.set_title("(a) Query Performance Regressions", fontweight='bold')
    ax.legend()
    
    # (b) Performance Distribution (Tail Risk)
    ax = axes[1]
    sns.kdeplot(res_df['bao_delta'], color=COLORS['Bao'], label='Bao-like', fill=True, ax=ax, alpha=0.3)
    sns.kdeplot(res_df['halo_delta'], color=COLORS['HALO'], label='HALO', fill=True, ax=ax, alpha=0.3)
    
    # Baseline
    ax.axvline(0, color='gray', linestyle='--', label='NATIVE Baseline (No Change)')
    ax.axvspan(0.1, 3.0, color='red', alpha=0.05, label='Regression Zone')
    
    ax.set_xlim(-1.5, 2.0)
    ax.set_xlabel("Query Execution Time Difference (Log Delta)")
    ax.set_title("(b) Performance Density (Focus on the 'Thick Tail')", fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_eval_bao_comparison.png'), dpi=150)
    plt.close()
    print(f"\nSaved comparison figure to {os.path.join(OUT_DIR, 'fig_eval_bao_comparison.png')}")

if __name__ == "__main__":
    run_bao_vs_halo_experiment()
