"""
HALO-R vs HALO-P Policy Comparison Visualization
=================================================
Proves the trade-off between Coverage (% of queries receiving hints)
and Safety (% of recommended hints that don't regress) for both policies.

Uses actual v4 sigma predictions to simulate both policy decisions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

PREDICTIONS_PATH = '/root/halo/results/v4/sigma_predictions_v4.parquet'
OUT_DIR = '/root/halo/assets'
os.makedirs(OUT_DIR, exist_ok=True)

def simulate_policies():
    """Simulate HALO-R and HALO-P decisions using v4 predictions."""
    print("Loading v4 predictions...")
    df = pd.read_parquet(PREDICTIONS_PATH)
    
    # Aggregate to query-hint level
    qh = df.groupby(['pair_label', 'query_id', 'hint_set']).agg(
        actual_delta_max=('delta_time', 'max'),
        actual_delta_mean=('delta_time', 'mean'),
        pred_mu_sum=('predicted_mu', 'sum'),
        pred_sigma_mean=('predicted_sigma', 'mean'),
        upper_bound_max=('upper_bound', 'max'),
        upper_bound_mean=('upper_bound', 'mean'),
        n_ops=('delta_time', 'count')
    ).reset_index()
    
    # Sweep across penalty thresholds to build the Pareto curve
    # HALO-R uses upper_bound threshold = 0.1 with high penalty
    # HALO-P uses upper_bound threshold = 0.1 with low penalty
    # We can generalize this by sweeping the upper_bound threshold
    
    thresholds = np.linspace(-0.5, 2.0, 50)
    
    results_sweep = []
    for thresh in thresholds:
        for pair in qh['pair_label'].unique():
            pdf = qh[qh['pair_label'] == pair]
            queries = pdf.groupby('query_id')
            
            total_q = len(queries)
            recommended = 0
            safe_recs = 0
            total_speedup = 0.0
            
            for qid, group in queries:
                # Filter hints where upper_bound_max <= threshold
                safe_hints = group[group['upper_bound_max'] <= thresh]
                
                if len(safe_hints) > 0:
                    # Pick best predicted hint
                    best = safe_hints.loc[safe_hints['pred_mu_sum'].idxmin()]
                    recommended += 1
                    actual = best['actual_delta_max']
                    
                    if actual <= 0.1:  # No significant regression
                        safe_recs += 1
                    
                    # Speedup: negative delta = faster
                    total_speedup += max(0, -actual)
                # else: NATIVE fallback
            
            coverage = recommended / max(total_q, 1) * 100
            safety = safe_recs / max(recommended, 1) * 100
            avg_speedup = total_speedup / max(total_q, 1)
            
            results_sweep.append({
                'threshold': thresh,
                'pair': pair,
                'coverage': coverage,
                'safety': safety,
                'avg_speedup': avg_speedup,
                'n_recommended': recommended,
                'n_safe': safe_recs,
                'n_total': total_q
            })
    
    return pd.DataFrame(results_sweep), qh

def generate_figure(sweep_df, qh_df):
    """Generate the dual-panel policy comparison figure."""
    
    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)
    
    # ──────────────────────────────────────────────────
    # Panel (a): Coverage vs Safety Pareto Curve
    # ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    
    # Average across all pairs for the sweep
    avg_sweep = sweep_df.groupby('threshold').agg(
        coverage=('coverage', 'mean'),
        safety=('safety', 'mean'),
        avg_speedup=('avg_speedup', 'mean')
    ).reset_index()
    
    # Plot the Pareto curve colored by threshold
    sc = ax1.scatter(avg_sweep['coverage'], avg_sweep['safety'], 
                     c=avg_sweep['threshold'], cmap='RdYlGn_r', 
                     s=60, edgecolors='gray', linewidth=0.5, zorder=5)
    ax1.plot(avg_sweep['coverage'], avg_sweep['safety'], '-', color='gray', alpha=0.3, zorder=1)
    
    # Mark HALO-R operating point (threshold ≈ 0.1)
    r_point = avg_sweep.iloc[(avg_sweep['threshold'] - 0.1).abs().argsort()[:1]]
    ax1.scatter(r_point['coverage'], r_point['safety'], 
                s=300, marker='*', color='#2e7d32', edgecolors='black', linewidth=1.5, zorder=10)
    ax1.annotate('HALO-R\n(Safety-First)', 
                xy=(float(r_point['coverage']), float(r_point['safety'])),
                xytext=(float(r_point['coverage']) + 8, float(r_point['safety']) - 5),
                fontsize=13, fontweight='bold', color='#2e7d32',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))
    
    # Mark HALO-P operating point (threshold ≈ 0.5)
    p_point = avg_sweep.iloc[(avg_sweep['threshold'] - 0.5).abs().argsort()[:1]]
    ax1.scatter(p_point['coverage'], p_point['safety'], 
                s=300, marker='D', color='#e65100', edgecolors='black', linewidth=1.5, zorder=10)
    ax1.annotate('HALO-P\n(Performance)', 
                xy=(float(p_point['coverage']), float(p_point['safety'])),
                xytext=(float(p_point['coverage']) + 5, float(p_point['safety']) + 3),
                fontsize=13, fontweight='bold', color='#e65100',
                arrowprops=dict(arrowstyle='->', color='#e65100', lw=2))
    
    # Mark Bao-like (no filter, threshold = infinity)
    bao_point = avg_sweep.iloc[-1:]  # largest threshold
    ax1.scatter(bao_point['coverage'], bao_point['safety'], 
                s=200, marker='X', color='#c62828', edgecolors='black', linewidth=1.5, zorder=10)
    ax1.annotate('Bao-like\n(No Filter)', 
                xy=(float(bao_point['coverage']), float(bao_point['safety'])),
                xytext=(float(bao_point['coverage']) - 15, float(bao_point['safety']) - 8),
                fontsize=11, fontweight='bold', color='#c62828',
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5))
    
    cb = plt.colorbar(sc, ax=ax1, shrink=0.7, label='Safety Threshold (σ upper bound)')
    
    ax1.set_xlabel('Coverage (%)\n(How many queries receive a hint recommendation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Safety Rate (%)\n(% of recommendations that don\'t regress)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Coverage vs Safety Pareto Frontier', fontsize=15, fontweight='bold')
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(0, 105)
    ax1.axhline(90, color='green', linestyle=':', alpha=0.4, label='90% Safety Target')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)
    
    # ──────────────────────────────────────────────────
    # Panel (b): Per-Pair Policy Outcome Comparison
    # ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    
    # For each pair: show HALO-R vs HALO-P outcomes
    pairs = sorted(qh_df['pair_label'].unique())
    pair_labels = [p.replace('_vs_', '\n→\n') for p in pairs]
    
    r_thresh = 0.1
    p_thresh = 0.5
    
    pair_data = []
    for pair in pairs:
        pdf = qh_df[qh_df['pair_label'] == pair]
        queries = pdf.groupby('query_id')
        total_q = len(queries)
        
        for thresh, mode in [(r_thresh, 'HALO-R'), (p_thresh, 'HALO-P')]:
            rec = 0
            safe = 0
            for qid, group in queries:
                safe_hints = group[group['upper_bound_max'] <= thresh]
                if len(safe_hints) > 0:
                    best = safe_hints.loc[safe_hints['pred_mu_sum'].idxmin()]
                    rec += 1
                    if best['actual_delta_max'] <= 0.1:
                        safe += 1
            
            pair_data.append({
                'pair': pair,
                'mode': mode,
                'coverage': rec / max(total_q, 1) * 100,
                'safety': safe / max(rec, 1) * 100,
                'n_rec': rec
            })
    
    pair_df = pd.DataFrame(pair_data)
    
    x = np.arange(len(pairs))
    width = 0.35
    
    r_cov = pair_df[pair_df['mode'] == 'HALO-R']['coverage'].values
    p_cov = pair_df[pair_df['mode'] == 'HALO-P']['coverage'].values
    
    bars_r = ax2.bar(x - width/2, r_cov, width, color='#2e7d32', alpha=0.85, label='HALO-R', edgecolor='white')
    bars_p = ax2.bar(x + width/2, p_cov, width, color='#e65100', alpha=0.85, label='HALO-P', edgecolor='white')
    
    for b, v in zip(bars_r, r_cov):
        if v > 0:
            ax2.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#2e7d32')
    for b, v in zip(bars_p, p_cov):
        if v > 0:
            ax2.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#e65100')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_labels, fontsize=9)
    ax2.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Coverage by HW Pair', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.2)
    
    # ──────────────────────────────────────────────────
    # Panel (c): Safety Rate per pair
    # ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    
    r_safe = pair_df[pair_df['mode'] == 'HALO-R']['safety'].values
    p_safe = pair_df[pair_df['mode'] == 'HALO-P']['safety'].values
    
    bars_r2 = ax3.bar(x - width/2, r_safe, width, color='#2e7d32', alpha=0.85, label='HALO-R', edgecolor='white')
    bars_p2 = ax3.bar(x + width/2, p_safe, width, color='#e65100', alpha=0.85, label='HALO-P', edgecolor='white')
    
    for b, v in zip(bars_r2, r_safe):
        if v > 0:
            ax3.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#2e7d32')
    for b, v in zip(bars_p2, p_safe):
        if v > 0:
            ax3.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#e65100')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(pair_labels, fontsize=9)
    ax3.set_ylabel('Safety Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Safety Rate by HW Pair', fontsize=15, fontweight='bold')
    ax3.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% Target')
    ax3.legend(fontsize=11)
    ax3.set_ylim(0, 110)
    ax3.grid(axis='y', alpha=0.2)
    
    fig.suptitle('HALO Policy Comparison: Safety-First (R) vs Performance (P)', 
                 fontsize=18, fontweight='bold', y=1.03)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_policy_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved policy comparison to {out_path}")

if __name__ == "__main__":
    sweep_df, qh_df = simulate_policies()
    generate_figure(sweep_df, qh_df)
