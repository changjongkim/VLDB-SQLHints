import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PREDICTIONS_PATH = '/root/halo/results/v4/sigma_predictions_v4.parquet'
OUT_DIR = '/root/halo/assets'
os.makedirs(OUT_DIR, exist_ok=True)

def fig_bao_vs_halo_4x4_heatmap():
    print("Generating 4x4 Transfer Heatmap (Train -> Test) modeled after fig4_bao_transfer...")
    df = pd.read_parquet(PREDICTIONS_PATH)
    
    # query-hint level aggregation
    qh_df = df.groupby(['env1', 'env2', 'query_id', 'hint_set']).agg(
        actual_delta_max=('delta_time', 'max'),
        pred_mu_sum=('predicted_mu', 'sum'),
        upper_bound_max=('upper_bound', 'max')
    ).reset_index()

    # Calculate Regression Rates for each (Train, Test) pair
    # For Bao: pick lowest mu. Regression if delta_time > 0.1
    # For HALO: pick lowest mu if upper_bound <= 0.1, else fallback (delta=0).
    
    envs = ['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA']
    
    bao_matrix = pd.DataFrame(index=envs, columns=envs, dtype=float)
    halo_matrix = pd.DataFrame(index=envs, columns=envs, dtype=float)
    
    for train_env in envs:
        for test_env in envs:
            if train_env == test_env:
                # Home turf is typically safer, we pretend it's evaluated for completeness
                bao_matrix.loc[train_env, test_env] = 0.0
                halo_matrix.loc[train_env, test_env] = 0.0
                continue
                
            pdf_raw = qh_df[(qh_df['env1'] == train_env) & (qh_df['env2'] == test_env)]
            if len(pdf_raw) == 0:
                # Some LOGO combinations might reverse the pairing direction (e.g., A_NVMe vs B_SATA is captured as one pair)
                # If direct train->test is missing, look for the reverse in the pair mapping or leave as NaN
                # The parquet has 'env1' and 'env2'. Let's check both ways.
                pdf_raw = qh_df[(qh_df['env1'] == test_env) & (qh_df['env2'] == train_env)]
                
            if len(pdf_raw) == 0:
                bao_matrix.loc[train_env, test_env] = np.nan
                halo_matrix.loc[train_env, test_env] = np.nan
                continue
                
            grouped = pdf_raw.groupby('query_id')
            total_queries = len(grouped)
            bao_regs = 0
            halo_regs = 0
            
            for qid, group in grouped:
                # BAO
                bao_ch = group.loc[group['pred_mu_sum'].idxmin()]
                if bao_ch['actual_delta_max'] > 0.1: # 10% regression threshold typical for these plots
                    bao_regs += 1
                
                # HALO
                safe_hints = group[group['upper_bound_max'] <= 0.1]
                if len(safe_hints) > 0:
                    halo_ch = safe_hints.loc[safe_hints['pred_mu_sum'].idxmin()]
                    if halo_ch['actual_delta_max'] > 0.1:
                        halo_regs += 1
                # If no safe hints, HALO does 0 regression
            
            bao_matrix.loc[train_env, test_env] = (bao_regs / total_queries) * 100
            halo_matrix.loc[train_env, test_env] = (halo_regs / total_queries) * 100

    # Fill diagonal nicely
    np.fill_diagonal(bao_matrix.values, np.nan)
    np.fill_diagonal(halo_matrix.values, np.nan)

    # Plotting style just like fig4_bao_transfer.png
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Hardware Transfer Robustness (Train → Test)', fontsize=22, fontweight='bold', y=1.05)

    # Format env names for display 
    display_envs = [e.replace('_', '\n') for e in envs]
    
    # Colormap: White (0) to Dark Red (High)
    cmap = LinearSegmentedColormap.from_list('reg_cmap', ['#fff5f0', '#fcbba1', '#fb6a4a', '#cb181d', '#67000d'])
    
    vmax = max(np.nanmax(bao_matrix.values), np.nanmax(halo_matrix.values)) + 5

    # (a) BAO Regression Rate
    sns.heatmap(bao_matrix, annot=True, fmt='.1f', cmap=cmap, vmin=0, vmax=vmax,
                cbar_kws={'label': 'Regression Rate (%)'}, 
                linewidths=0.5, linecolor='gray', ax=axes[0], 
                annot_kws={"size": 15, "weight": "bold"})
    
    # Format texts with %
    for t in axes[0].texts:
        if t.get_text() != 'nan':
            t.set_text(t.get_text() + "%")
        else:
            t.set_text('-')
            
    axes[0].set_title('(a) BAO Regression Rate (Greedy)', fontsize=18, fontweight='bold', pad=15)
    axes[0].set_xticklabels(display_envs, fontsize=14)
    axes[0].set_yticklabels(display_envs, fontsize=14, rotation=0)
    axes[0].set_xlabel('Test Environment', fontsize=16)
    axes[0].set_ylabel('Train Environment', fontsize=16)

    # (b) HALO Regression Rate
    sns.heatmap(halo_matrix, annot=True, fmt='.1f', cmap='Greens', vmin=0, vmax=vmax/2,
                cbar_kws={'label': 'Regression Rate (%)'}, 
                linewidths=0.5, linecolor='gray', ax=axes[1], 
                annot_kws={"size": 15, "weight": "bold", "color": "#1b5e20"})
    
    for t in axes[1].texts:
        if t.get_text() != 'nan':
            t.set_text(t.get_text() + "%")
        else:
            t.set_text('-')

    axes[1].set_title('(b) HALO v4 Regression Rate (Safety-Gated)', fontsize=18, fontweight='bold', pad=15)
    axes[1].set_xticklabels(display_envs, fontsize=14)
    axes[1].set_yticklabels(display_envs, fontsize=14, rotation=0)
    axes[1].set_xlabel('Test Environment', fontsize=16)
    axes[1].set_ylabel('Train Environment', fontsize=16)

    # Clean spines
    for ax in axes:
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    # Overwrite the old figure name so it seamlessly drops into the README
    save_path = os.path.join(OUT_DIR, 'fig_eval_bao_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Generated 4x4 matrix Heatmap to {save_path}")

if __name__ == "__main__":
    fig_bao_vs_halo_4x4_heatmap()
