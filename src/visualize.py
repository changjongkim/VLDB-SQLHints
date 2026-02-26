# -*- coding: utf-8 -*-
"""
Halo: Publication-quality Visualizations for Key Results
Generates figures for E1 (HPP), E5 (Head-to-Head), E8 (Sigma), and model performance.
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#2563EB',
    'red': '#DC2626',
    'green': '#16A34A',
    'orange': '#EA580C',
    'purple': '#9333EA',
    'teal': '#0D9488',
    'gray': '#6B7280',
    'pink': '#DB2777',
}

OUTPUT_DIR = '/root/halo/results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Figure 1: HPP Existence (E1) - Portability Score Distribution
# ============================================================

def fig1_hpp_portability():
    """Show that hint effects don't transfer across hardware."""
    with open('/root/halo/results/experiment_results.json') as f:
        results = json.load(f)

    # Reconstruct portability data from raw queries
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')

    baselines = df_q[df_q['hint_set'] == 'baseline'].copy()
    hinted = df_q[df_q['hint_set'] != 'baseline'].copy()

    baseline_times = baselines.set_index(['query_id', 'env_id'])['execution_time_s'].to_dict()

    speedups = []
    for _, row in hinted.iterrows():
        bkey = (row['query_id'], row['env_id'])
        if bkey in baseline_times and baseline_times[bkey] > 0:
            s = baseline_times[bkey] / max(row['execution_time_s'], 0.001)
            speedups.append({
                'query_id': row['query_id'],
                'hint_set': row['hint_set'],
                'hint_variant': row['hint_variant'],
                'env_id': row['env_id'],
                'speedup': s,
            })

    df_sp = pd.DataFrame(speedups)
    envs = sorted(df_sp['env_id'].unique())

    # Compute portability scores for all env pairs
    all_p = []
    for i, e1 in enumerate(envs):
        for e2 in envs[i+1:]:
            s1 = df_sp[df_sp['env_id'] == e1].set_index(
                ['query_id', 'hint_set', 'hint_variant'])['speedup']
            s2 = df_sp[df_sp['env_id'] == e2].set_index(
                ['query_id', 'hint_set', 'hint_variant'])['speedup']
            common = s1.index.intersection(s2.index)
            for key in common:
                v1 = s1.loc[key]
                v2 = s2.loc[key]
                if isinstance(v1, pd.Series): v1 = v1.iloc[0]
                if isinstance(v2, pd.Series): v2 = v2.iloc[0]
                if v1 > 0.01:
                    all_p.append({
                        'portability': v2 / v1,
                        'env_pair': f"{e1} vs {e2}",
                        'speedup_e1': v1, 'speedup_e2': v2,
                    })

    df_p = pd.DataFrame(all_p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of portability scores
    ax = axes[0]
    p_vals = df_p['portability'].clip(0, 3)
    ax.hist(p_vals, bins=50, color=COLORS['blue'], alpha=0.7, edgecolor='white',
            linewidth=0.5)
    ax.axvline(x=1.0, color=COLORS['red'], linewidth=2, linestyle='--',
               label='Perfect Portability (P=1.0)')
    ax.axvline(x=0.7, color=COLORS['orange'], linewidth=2, linestyle=':',
               label='Effect Lost (P=0.7)')

    p_lt07 = (df_p['portability'] < 0.7).mean() * 100
    ax.fill_betweenx([0, ax.get_ylim()[1] or 200], 0, 0.7,
                     alpha=0.15, color=COLORS['red'])
    ax.text(0.35, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 150,
            f'{p_lt07:.1f}%\neffect lost',
            ha='center', fontsize=13, color=COLORS['red'], fontweight='bold')

    ax.set_xlabel('Portability Score (P = S_env2 / S_env1)')
    ax.set_ylabel('Number of Hint-Query Pairs')
    ax.set_title('(a) Hint Portability Score Distribution')
    ax.legend(loc='upper right', fontsize=10)

    # Right: Scatter - speedup env1 vs env2
    ax = axes[1]
    ax.scatter(df_p['speedup_e1'].clip(0.1, 10),
               df_p['speedup_e2'].clip(0.1, 10),
               alpha=0.25, s=15, c=COLORS['blue'], edgecolors='none')
    lim = max(df_p['speedup_e1'].clip(0.1, 10).max(),
              df_p['speedup_e2'].clip(0.1, 10).max())
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, alpha=0.7,
            label='Perfect Transfer')
    ax.plot([0, lim], [0, lim * 0.7], color=COLORS['orange'], linestyle=':',
            linewidth=1.5, alpha=0.7, label='30% Degradation')
    ax.set_xlabel('Speedup in Environment 1')
    ax.set_ylabel('Speedup in Environment 2')
    ax.set_title('(b) Hint Effect: Env1 vs Env2')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0.1, min(lim + 0.5, 10))
    ax.set_ylim(0.1, min(lim + 0.5, 10))

    fig.suptitle('E1: The Hint Portability Problem (HPP)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig1_hpp_portability.png')
    plt.close()
    print(f"Saved: fig1_hpp_portability.png")


# ============================================================
# Figure 2: Operator-Level Sigma Heatmap (E8)
# ============================================================

def fig2_sigma_heatmap():
    """Heatmap of sigma values per operator type."""
    with open('/root/halo/results/sigma_results_v2.json') as f:
        data = json.load(f)

    op_sigmas = data['operator_sigma']
    # Sort by max sensitivity
    op_sigmas = sorted(op_sigmas,
                       key=lambda x: max(x['sigma_storage'], x['sigma_compute']),
                       reverse=True)

    # Filter to operators with enough samples
    op_sigmas = [o for o in op_sigmas if o['count'] >= 20]

    labels = [o['op_type'].replace('_', ' ') for o in op_sigmas]
    sig_s = [o['sigma_storage'] for o in op_sigmas]
    sig_c = [o['sigma_compute'] for o in op_sigmas]
    counts = [o['count'] for o in op_sigmas]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    bars1 = ax.barh(y_pos - bar_height/2, sig_s, bar_height,
                    color=COLORS['teal'], alpha=0.85, label='$\\sigma_{storage}$',
                    edgecolor='white', linewidth=0.5)
    bars2 = ax.barh(y_pos + bar_height/2, sig_c, bar_height,
                    color=COLORS['purple'], alpha=0.85, label='$\\sigma_{compute}$',
                    edgecolor='white', linewidth=0.5)

    # Add count annotations
    for i, (s, c, n) in enumerate(zip(sig_s, sig_c, counts)):
        max_val = max(s, c)
        ax.text(max_val + 0.05, i, f'n={n:,}', va='center', fontsize=9,
                color=COLORS['gray'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Hardware Sensitivity ($\\sigma$)', fontsize=14)
    ax.set_title('E8: Operator-Level Hardware Sensitivity ($\\sigma$)', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=13)

    # Highlight physical interpretation
    ax.axvline(x=0.5, color=COLORS['gray'], linewidth=1, linestyle=':', alpha=0.5)
    ax.text(0.52, len(labels) - 0.5, 'Sensitive\nthreshold', fontsize=9,
            color=COLORS['gray'], va='top')

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig2_sigma_heatmap.png')
    plt.close()
    print(f"Saved: fig2_sigma_heatmap.png")


# ============================================================
# Figure 3: Head-to-Head Performance (E5)
# ============================================================

def fig3_head_to_head():
    """Bar chart comparing strategies across environments."""
    with open('/root/halo/results/experiment_results.json') as f:
        data = json.load(f)

    e5 = data['E5_head_to_head']
    envs = sorted(e5.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(envs))
    width = 0.25

    b1_vals = [e5[e]['B1_native_total'] for e in envs]
    b4_vals = [e5[e]['B4_total'] for e in envs]
    b4_hints = [e5[e]['B4_best_hint_set'] for e in envs]
    b4_speedups = [e5[e]['B4_speedup'] for e in envs]

    bars1 = ax.bar(x - width/2, b1_vals, width, label='B1: Native Optimizer',
                   color=COLORS['gray'], alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, b4_vals, width, label='B4: Best Hint Set (Per-Env Optimal)',
                   color=COLORS['green'], alpha=0.8, edgecolor='white')

    # Add speedup labels
    for i, (b1, b4, hint, sp) in enumerate(zip(b1_vals, b4_vals, b4_hints, b4_speedups)):
        if sp > 1.0:
            ax.annotate(f'{sp:.2f}x\n({hint})',
                       xy=(i + width/2, b4), xytext=(i + width/2, b4 + 200),
                       ha='center', fontsize=11, fontweight='bold',
                       color=COLORS['green'],
                       arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5))
        else:
            ax.text(i + width/2, b4 + 100, f'{hint}', ha='center', fontsize=10)

    # Highlight: different best hints per env
    ax.set_xticks(x)
    env_labels = []
    for e in envs:
        parts = e.split('_')
        env_labels.append(f"Server {parts[0]}\n{parts[1]}")
    ax.set_xticklabels(env_labels, fontsize=13)

    ax.set_ylabel('Total Walltime (seconds)', fontsize=14)
    ax.set_title('E5: Best Hint Set Varies by Hardware Environment', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')

    # Add annotation box
    ax.annotate('Key Finding: Optimal hint set\nchanges across environments\n(hint05 on A, hint02 on B)',
               xy=(0.98, 0.98), xycoords='axes fraction',
               ha='right', va='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor=COLORS['orange'], alpha=0.9))

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig3_head_to_head.png')
    plt.close()
    print(f"Saved: fig3_head_to_head.png")


# ============================================================
# Figure 4: BAO Transfer Failure (E2)
# ============================================================

def fig4_bao_transfer():
    """Show BAO performance degrades when transferring across hardware."""
    with open('/root/halo/results/experiment_results.json') as f:
        data = json.load(f)

    e2 = data['E2_BAO_transfer']

    # Build matrix
    envs = ['A_NVMe', 'A_SATA', 'B_NVMe', 'B_SATA']
    available_envs = []
    for e in envs:
        if any(k.startswith(e + '->') for k in e2.keys()):
            available_envs.append(e)

    n = len(available_envs)
    speedup_matrix = np.zeros((n, n))
    regression_matrix = np.zeros((n, n))

    for i, train_env in enumerate(available_envs):
        for j, test_env in enumerate(available_envs):
            key = f"{train_env}->{test_env}"
            if key in e2:
                speedup_matrix[i, j] = e2[key]['overall_speedup']
                regression_matrix[i, j] = e2[key]['regression_rate'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Speedup heatmap
    ax = axes[0]
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0.8, vmax=max(1.3, speedup_matrix.max()))
    for i in range(n):
        for j in range(n):
            val = speedup_matrix[i, j]
            color = 'white' if val > 1.1 or val < 0.9 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)

    env_short = [e.replace('_', '\n') for e in available_envs]
    ax.set_xticks(range(n))
    ax.set_xticklabels(env_short, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(env_short, fontsize=11)
    ax.set_xlabel('Test Environment', fontsize=13)
    ax.set_ylabel('Train Environment', fontsize=13)
    ax.set_title('(a) BAO Speedup (Train → Test)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Speedup')

    # Regression rate heatmap
    ax = axes[1]
    im2 = ax.imshow(regression_matrix, cmap='Reds', aspect='auto',
                    vmin=0, vmax=max(5, regression_matrix.max()))
    for i in range(n):
        for j in range(n):
            val = regression_matrix[i, j]
            color = 'white' if val > 2 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(env_short, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(env_short, fontsize=11)
    ax.set_xlabel('Test Environment', fontsize=13)
    ax.set_ylabel('Train Environment', fontsize=13)
    ax.set_title('(b) Regression Rate (Train → Test)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='Regression %')

    fig.suptitle('E2: BAO-inspired Model Transfer Failure', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig4_bao_transfer.png')
    plt.close()
    print(f"Saved: fig4_bao_transfer.png")


# ============================================================
# Figure 5: Feature Importance + Model Performance (E3)
# ============================================================

def fig5_model_analysis():
    """Feature importance and prediction scatter plot."""
    with open('/root/halo/results/sigma_results_v2.json') as f:
        data = json.load(f)

    df_pred = pd.read_parquet('/root/halo/results/sigma_predictions_v2.parquet')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Feature importance (top 12)
    ax = axes[0]
    feat_imp = data['feature_importance']
    items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:12]
    names = [x[0].replace('is_', '').replace('_', ' ') for x in items]
    values = [x[1] for x in items]

    colors_list = []
    for name in [x[0] for x in items]:
        if 'changed' in name:
            colors_list.append(COLORS['red'])
        elif name.startswith('is_'):
            colors_list.append(COLORS['blue'])
        elif 'log_' in name or 'row' in name:
            colors_list.append(COLORS['teal'])
        else:
            colors_list.append(COLORS['gray'])

    bars = ax.barh(range(len(names)), values, color=colors_list, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=13)
    ax.set_title('(a) Top Features for $\\sigma$ Prediction', fontsize=14, fontweight='bold')

    # Legend for color coding
    legend_patches = [
        mpatches.Patch(color=COLORS['red'], label='HW Factor'),
        mpatches.Patch(color=COLORS['blue'], label='Op Type'),
        mpatches.Patch(color=COLORS['teal'], label='Cardinality'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)

    # Right: Predicted vs Actual delta_time
    ax = axes[1]
    actual = df_pred['delta_time'].values
    predicted = df_pred['predicted_delta'].values

    # Color by pair type
    is_cross = df_pred['pair_label'].str.contains('A_') & df_pred['pair_label'].str.contains('B_')

    ax.scatter(actual[~is_cross], predicted[~is_cross],
               alpha=0.15, s=8, c=COLORS['blue'], label='Same-server', edgecolors='none')
    ax.scatter(actual[is_cross], predicted[is_cross],
               alpha=0.4, s=15, c=COLORS['red'], label='Cross-server', edgecolors='none')

    lim = 3
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Actual $\\Delta_{time}$ (log ratio)', fontsize=13)
    ax.set_ylabel('Predicted $\\Delta_{time}$', fontsize=13)
    ax.set_title(f'(b) $\\sigma$ Model: r={data["pearson_r"]:.3f}, RMSE={data["overall_rmse"]:.3f}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig5_model_analysis.png')
    plt.close()
    print(f"Saved: fig5_model_analysis.png")


# ============================================================
# Figure 6: Plan Diff Summary
# ============================================================

def fig6_plan_diff():
    """Show what changes between baseline and hinted plans."""
    df_diffs = pd.read_parquet('/root/halo/data/plan_diffs.parquet')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Change type distribution
    ax = axes[0]
    ct = df_diffs['change_type'].value_counts()
    colors_pie = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]
    wedges, texts, autotexts = ax.pie(
        ct.values, labels=ct.index, autopct='%1.1f%%',
        colors=colors_pie[:len(ct)], startangle=90,
        textprops={'fontsize': 12},
        pctdistance=0.8
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax.set_title(f'(a) Plan Diff Types (N={len(df_diffs):,})', fontsize=14, fontweight='bold')

    # Right: Time impact of method changes
    ax = axes[1]
    method_changes = df_diffs[df_diffs['change_type'] == 'method_change'].copy()
    if len(method_changes) > 0:
        # Show top transitions
        method_changes['transition'] = (
            method_changes['op_type_baseline'] + '\n→ ' +
            method_changes['op_type_hinted']
        )
        top_transitions = method_changes['transition'].value_counts().head(8)

        colors_bar = [COLORS['blue'], COLORS['teal'], COLORS['purple'],
                     COLORS['orange'], COLORS['green'], COLORS['red'],
                     COLORS['pink'], COLORS['gray']]

        bars = ax.barh(range(len(top_transitions)), top_transitions.values,
                      color=colors_bar[:len(top_transitions)], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(top_transitions)))
        ax.set_yticklabels(top_transitions.index, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Occurences', fontsize=13)
        ax.set_title(f'(b) Top Operator Method Changes (N={len(method_changes):,})',
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig6_plan_diff.png')
    plt.close()
    print(f"Saved: fig6_plan_diff.png")


# ============================================================
# Figure 7: Summary Dashboard
# ============================================================

def fig7_summary_dashboard():
    """Single dashboard summarizing all key findings."""
    with open('/root/halo/results/experiment_results.json') as f:
        exp = json.load(f)
    with open('/root/halo/results/sigma_results_v2.json') as f:
        sigma = json.load(f)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── (1) HPP Summary ──
    ax = fig.add_subplot(gs[0, 0])
    e1 = exp['E1_HPP']
    categories = ['P < 0.7\n(Lost)', '0.7 ≤ P ≤ 2.0\n(Portable)', 'P > 2.0\n(Amplified)']
    values = [e1['p_lt_07_pct'], 100 - e1['p_lt_07_pct'] - e1['p_gt_2_pct'], e1['p_gt_2_pct']]
    colors = [COLORS['red'], COLORS['green'], COLORS['blue']]
    ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white')
    for i, v in enumerate(values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Hints')
    ax.set_title('E1: HPP Severity', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)

    # ── (2) Sigma model metrics ──
    ax = fig.add_subplot(gs[0, 1])
    metrics = ['Pearson r', 'Dir. Acc', 'RMSE']
    v1_vals = [0.314, 82.7, 1.966]
    v2_vals = [sigma['pearson_r'], sigma['direction_accuracy'], sigma['overall_rmse']]

    x = np.arange(len(metrics))
    w = 0.3
    ax.bar(x - w/2, v1_vals, w, label='V1 (Baseline)', color=COLORS['gray'], alpha=0.7)
    ax.bar(x + w/2, v2_vals, w, label='V2 (Improved)', color=COLORS['blue'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title('E3: $\\sigma$ Model Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # ── (3) Best hint per env ──
    ax = fig.add_subplot(gs[0, 2])
    e5 = exp['E5_head_to_head']
    env_names = []
    best_hints = []
    speedups = []
    for env in sorted(e5.keys()):
        if e5[env]['B4_speedup'] > 1.0:
            parts = env.split('_')
            env_names.append(f"{parts[0]}\n{parts[1]}")
            best_hints.append(e5[env]['B4_best_hint_set'])
            speedups.append(e5[env]['B4_speedup'])

    hint_colors = {'hint01': COLORS['blue'], 'hint02': COLORS['green'],
                   'hint04': COLORS['orange'], 'hint05': COLORS['purple']}
    bar_colors = [hint_colors.get(h, COLORS['gray']) for h in best_hints]

    bars = ax.bar(env_names, speedups, color=bar_colors, alpha=0.85, edgecolor='white')
    for i, (sp, hint) in enumerate(zip(speedups, best_hints)):
        ax.text(i, sp + 0.02, f'{sp:.2f}x\n{hint}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_ylabel('Speedup vs Native')
    ax.set_title('E5: Best Hint per Env', fontsize=14, fontweight='bold')
    ax.set_ylim(0.9, max(speedups) + 0.2)

    # ── (4) Top-5 sensitive operators ──
    ax = fig.add_subplot(gs[1, 0])
    op_stats = sigma['operator_sigma']
    op_stats = sorted(op_stats, key=lambda x: max(x['sigma_storage'], x['sigma_compute']), reverse=True)
    top5 = [o for o in op_stats if o['count'] >= 20][:5]

    y = np.arange(len(top5))
    h = 0.35
    ax.barh(y - h/2, [o['sigma_storage'] for o in top5], h,
            color=COLORS['teal'], alpha=0.85, label='$\\sigma_S$')
    ax.barh(y + h/2, [o['sigma_compute'] for o in top5], h,
            color=COLORS['purple'], alpha=0.85, label='$\\sigma_C$')
    ax.set_yticks(y)
    ax.set_yticklabels([o['op_type'].replace('_', ' ') for o in top5], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('$\\sigma$')
    ax.set_title('E8: Most Sensitive Ops', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # ── (5) Feature importance top 6 ──
    ax = fig.add_subplot(gs[1, 1])
    feat = sigma['feature_importance']
    items = sorted(feat.items(), key=lambda x: x[1], reverse=True)[:6]
    names = [x[0].replace('is_', '').replace('_', ' ').replace('log ', 'log\n') for x in items]
    vals = [x[1] for x in items]
    ax.barh(range(len(names)), vals, color=COLORS['blue'], alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')

    # ── (6) Key numbers ──
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    stats_text = (
        f"Halo: Key Numbers\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Operator Instances:  98,145\n"
        f"Query Results:        5,151\n"
        f"Environments:              4\n"
        f"Benchmarks:         JOB + TPC-H\n"
        f"Operator Types:           19\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"HPP Rate (P<0.7):    33.3%\n"
        f"$\\sigma$ Pearson r:     {sigma['pearson_r']:.3f}\n"
        f"$\\sigma$ RMSE:           {sigma['overall_rmse']:.3f}\n"
        f"Direction Acc:       {sigma['direction_accuracy']:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Selective: 0 regressions"
    )
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           fontsize=13, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow',
                    edgecolor=COLORS['orange'], alpha=0.9))

    fig.suptitle('Halo: Hardware-Aware Learned Optimizer (HALO) — Results Summary',
                fontsize=18, fontweight='bold', y=1.01)

    fig.savefig(f'{OUTPUT_DIR}/fig7_summary_dashboard.png')
    plt.close()
    print(f"Saved: fig7_summary_dashboard.png")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Generating Halo Figures...")
    fig1_hpp_portability()
    fig2_sigma_heatmap()
    fig3_head_to_head()
    fig4_bao_transfer()
    fig5_model_analysis()
    fig6_plan_diff()
    fig7_summary_dashboard()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("Files:", os.listdir(OUTPUT_DIR))
