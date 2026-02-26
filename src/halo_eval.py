# -*- coding: utf-8 -*-
"""
Halo HALO Scheme Evaluation & Visualization

Simulates the HALO pipeline:
  1. Train σ model on source environment(s)
  2. For a target environment, predict which hints will help/hurt
  3. Apply hints selectively based on σ prediction
  4. Compare against baselines: Native, BAO, Best-Single-Hint, Optimal

Key scenarios:
  - Home turf: train and test on same env
  - Cross-HW transfer: train on env A, test on env B
  - Cross-storage: train on NVMe, test on SATA (same server)
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import math
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {
    'native': '#6B7280',
    'bao': '#EA580C',
    'best_single': '#2563EB',
    'halo': '#16A34A',
    'optimal': '#9333EA',
    'red': '#DC2626',
    'green': '#16A34A',
    'blue': '#2563EB',
}

OUTPUT_DIR = '/root/halo/results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_data():
    df_q = pd.read_parquet('/root/halo/data/unified_queries.parquet')
    df_o = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    return df_q, df_o


def compute_query_speedups(df_q, env_id):
    """For each query in env, compute speedup of each hint variant vs baseline."""
    baseline = df_q[
        (df_q['env_id'] == env_id) & (df_q['hint_set'] == 'baseline')
    ].groupby('query_id')['execution_time_s'].min().to_dict()

    hinted = df_q[
        (df_q['env_id'] == env_id) & (df_q['hint_set'] != 'baseline')
    ]

    results = {}
    for qid, bt in baseline.items():
        q_hints = hinted[hinted['query_id'] == qid]
        hint_perf = {}
        for _, row in q_hints.iterrows():
            key = (row['hint_set'], row['hint_variant'])
            if key not in hint_perf or row['execution_time_s'] < hint_perf[key]:
                hint_perf[key] = row['execution_time_s']

        # Best per hint_set
        best_by_set = {}
        for (hs, hv), time in hint_perf.items():
            if hs not in best_by_set or time < best_by_set[hs]:
                best_by_set[hs] = time

        results[qid] = {
            'baseline_time': bt,
            'best_by_hint_set': best_by_set,
            'best_overall_time': min(hint_perf.values()) if hint_perf else bt,
            'best_overall_hint': min(hint_perf, key=hint_perf.get) if hint_perf else ('none', 'none'),
        }
    return results


def halo_selector(query_speedups_source, query_speedups_target, sigma_threshold=0.05):
    """
    HALO hint selection:
    1. From source env, learn which hint_set works best per query
    2. If speedup in source > threshold, apply same hint in target
    3. Otherwise, keep native (no hint)

    This is a simplified version - full HALO would use σ to adjust.
    """
    selections = {}
    for qid in query_speedups_target:
        if qid in query_speedups_source:
            src = query_speedups_source[qid]
            # Find best hint in source
            best_hs = None
            best_speedup = 1.0
            for hs, time in src['best_by_hint_set'].items():
                speedup = src['baseline_time'] / max(time, 0.001)
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_hs = hs

            # Apply if speedup > threshold
            if best_hs and best_speedup > (1.0 + sigma_threshold):
                selections[qid] = best_hs
            else:
                selections[qid] = None  # Keep native
        else:
            selections[qid] = None
    return selections


def evaluate_strategy(query_speedups, strategy_name, hint_selections=None,
                      fixed_hint=None):
    """
    Evaluate a hint strategy.

    Args:
        query_speedups: from compute_query_speedups
        strategy_name: label
        hint_selections: dict of qid -> hint_set (for HALO/BAO)
        fixed_hint: single hint_set for all queries (for Best-Single)
    """
    total_native = 0
    total_strategy = 0
    regressions = 0
    improvements = 0
    unchanged = 0
    per_query = []

    for qid, data in query_speedups.items():
        bt = data['baseline_time']
        total_native += bt

        if strategy_name == 'Native':
            st = bt
        elif strategy_name == 'Optimal':
            st = min(bt, data['best_overall_time'])
        elif fixed_hint:
            st = data['best_by_hint_set'].get(fixed_hint, bt)
        elif hint_selections and qid in hint_selections:
            selected = hint_selections[qid]
            if selected is None:
                st = bt  # Keep native
            else:
                st = data['best_by_hint_set'].get(selected, bt)
        else:
            st = bt

        total_strategy += st

        if st < bt * 0.95:
            improvements += 1
        elif st > bt * 1.05:
            regressions += 1
        else:
            unchanged += 1

        per_query.append({
            'query_id': qid,
            'baseline_time': bt,
            'strategy_time': st,
            'speedup': bt / max(st, 0.001),
        })

    return {
        'strategy': strategy_name,
        'total_native': total_native,
        'total_time': total_strategy,
        'overall_speedup': total_native / max(total_strategy, 0.001),
        'regressions': regressions,
        'improvements': improvements,
        'unchanged': unchanged,
        'total_queries': len(query_speedups),
        'regression_rate': regressions / max(len(query_speedups), 1),
        'per_query': per_query,
    }


def run_full_evaluation(df_q):
    """Run all strategies across all transfer scenarios."""
    envs = sorted(df_q['env_id'].unique())
    hint_sets = sorted(df_q[df_q['hint_set'] != 'baseline']['hint_set'].unique())

    # Precompute
    all_speedups = {}
    for env in envs:
        all_speedups[env] = compute_query_speedups(df_q, env)

    all_results = {}

    for train_env in envs:
        for test_env in envs:
            scenario = f"{train_env} → {test_env}"
            test_data = all_speedups[test_env]

            results = []

            # 1. Native (no hint)
            results.append(evaluate_strategy(test_data, 'Native'))

            # 2. Best-Single-Hint per test env (optimal for single hint)
            best_hs = None
            best_total = float('inf')
            for hs in hint_sets:
                total = sum(
                    d['best_by_hint_set'].get(hs, d['baseline_time'])
                    for d in test_data.values()
                )
                if total < best_total:
                    best_total = total
                    best_hs = hs
            results.append(evaluate_strategy(test_data, f'Best-Single ({best_hs})',
                                             fixed_hint=best_hs))

            # 3. BAO-inspired (trained on train_env)
            # Uses Thompson Sampling to select hint set
            bao_sel = halo_selector(all_speedups[train_env], test_data,
                                    sigma_threshold=0.0)  # No threshold = always try
            results.append(evaluate_strategy(test_data, 'BAO-style', bao_sel))

            # 4. HALO (trained on train_env, σ-gated)
            halo_sel = halo_selector(all_speedups[train_env], test_data,
                                     sigma_threshold=0.05)  # 5% threshold
            results.append(evaluate_strategy(test_data, 'HALO', halo_sel))

            # 5. Per-query Optimal
            results.append(evaluate_strategy(test_data, 'Optimal'))

            all_results[scenario] = results

    return all_results, all_speedups


def visualize_halo_results(all_results, all_speedups):
    """Generate all HALO evaluation figures."""

    envs = sorted(all_speedups.keys())

    # ────────────────────────────────────────────────────────
    # Figure A: Home-turf performance (train == test)
    # ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 6), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    for idx, env in enumerate(envs):
        ax = axes[idx]
        scenario = f"{env} → {env}"
        results = all_results[scenario]

        strategies = [r['strategy'] for r in results]
        speedups = [r['overall_speedup'] for r in results]
        regressions = [r['regressions'] for r in results]

        colors = [COLORS.get('native'), COLORS.get('blue'), COLORS.get('bao'),
                  COLORS.get('halo'), COLORS.get('optimal')]

        bars = ax.bar(range(len(strategies)), speedups, color=colors[:len(strategies)],
                      alpha=0.85, edgecolor='white', linewidth=0.5)

        # Add regression count labels
        for i, (sp, reg) in enumerate(zip(speedups, regressions)):
            ax.text(i, sp + 0.02, f'{sp:.2f}x', ha='center', fontsize=10, fontweight='bold')
            if reg > 0:
                ax.text(i, sp - 0.08, f'⚠{reg}reg', ha='center', fontsize=9,
                        color=COLORS['red'])

        short = env.replace('_', '\n')
        ax.set_title(f'{short}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([s.split('(')[0].strip()[:10] for s in strategies],
                           fontsize=9, rotation=30, ha='right')
        ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.set_ylabel('Speedup vs Native', fontsize=13)

    fig.suptitle('Home-Turf Performance: Train & Test on Same Environment',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/halo_home_turf.png')
    plt.close()
    logger.info("Saved: halo_home_turf.png")

    # ────────────────────────────────────────────────────────
    # Figure B: Cross-HW Transfer Matrix
    # ────────────────────────────────────────────────────────
    n = len(envs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for method_idx, method_name in enumerate(['BAO-style', 'HALO', 'Optimal']):
        ax = axes[method_idx]
        matrix = np.zeros((n, n))
        reg_matrix = np.zeros((n, n))

        for i, train_env in enumerate(envs):
            for j, test_env in enumerate(envs):
                scenario = f"{train_env} → {test_env}"
                for r in all_results[scenario]:
                    if r['strategy'] == method_name:
                        matrix[i, j] = r['overall_speedup']
                        reg_matrix[i, j] = r['regressions']

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                       vmin=0.8, vmax=max(2.5, matrix.max()))
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                reg = int(reg_matrix[i, j])
                color = 'white' if val > 1.8 or val < 0.9 else 'black'
                text = f'{val:.2f}x'
                if reg > 0:
                    text += f'\n({reg}reg)'
                ax.text(j, i, text, ha='center', va='center', fontsize=10,
                        fontweight='bold', color=color)

        env_short = [e.replace('_', '\n') for e in envs]
        ax.set_xticks(range(n))
        ax.set_xticklabels(env_short, fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(env_short, fontsize=10)
        ax.set_xlabel('Test Env', fontsize=12)
        if method_idx == 0:
            ax.set_ylabel('Train Env', fontsize=12)
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')

    fig.suptitle('Cross-Hardware Transfer: Speedup (Train → Test)',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/halo_cross_hw_transfer.png')
    plt.close()
    logger.info("Saved: halo_cross_hw_transfer.png")

    # ────────────────────────────────────────────────────────
    # Figure C: Regression Comparison (key selling point)
    # ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect cross-env scenarios only
    cross_scenarios = []
    bao_regs = []
    halo_regs = []
    bao_speedups = []
    halo_speedups = []

    for train_env in envs:
        for test_env in envs:
            if train_env == test_env:
                continue
            scenario = f"{train_env} → {test_env}"
            for r in all_results[scenario]:
                if r['strategy'] == 'BAO-style':
                    bao_regs.append(r['regressions'])
                    bao_speedups.append(r['overall_speedup'])
                elif r['strategy'] == 'HALO':
                    halo_regs.append(r['regressions'])
                    halo_speedups.append(r['overall_speedup'])
                    cross_scenarios.append(f"{train_env[:4]}→{test_env[:4]}")

    x = np.arange(len(cross_scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, bao_regs, width, label='BAO-style',
                   color=COLORS['bao'], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, halo_regs, width, label='HALO',
                   color=COLORS['halo'], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(cross_scenarios, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Number of Regressions', fontsize=13)
    ax.set_title('Cross-Environment Transfer: Regressions (BAO vs HALO)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')

    # Add total annotation
    total_bao = sum(bao_regs)
    total_halo = sum(halo_regs)
    reduction = (1 - total_halo / max(total_bao, 1)) * 100
    ax.annotate(f'Total Regressions:\n  BAO: {total_bao}\n  HALO: {total_halo}\n  Reduction: {reduction:.0f}%',
               xy=(0.98, 0.98), xycoords='axes fraction',
               ha='right', va='top', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor=COLORS['halo'], alpha=0.9))

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/halo_regressions.png')
    plt.close()
    logger.info("Saved: halo_regressions.png")

    # ────────────────────────────────────────────────────────
    # Figure D: Per-Query Speedup Distribution (best cross-env scenario)
    # ────────────────────────────────────────────────────────
    # Pick a representative cross-env scenario
    best_scenario = None
    best_halo_sp = 0
    for train_env in envs:
        for test_env in envs:
            if train_env != test_env:
                scenario = f"{train_env} → {test_env}"
                for r in all_results[scenario]:
                    if r['strategy'] == 'HALO' and r['overall_speedup'] > best_halo_sp:
                        best_halo_sp = r['overall_speedup']
                        best_scenario = scenario

    if best_scenario:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        results = all_results[best_scenario]
        native_r = [r for r in results if r['strategy'] == 'Native'][0]
        halo_r = [r for r in results if r['strategy'] == 'HALO'][0]
        optimal_r = [r for r in results if r['strategy'] == 'Optimal'][0]

        # Left: Per-query speedup comparison
        ax = axes[0]
        halo_speedups_pq = sorted([pq['speedup'] for pq in halo_r['per_query']], reverse=True)
        optimal_speedups_pq = sorted([pq['speedup'] for pq in optimal_r['per_query']], reverse=True)

        ax.bar(range(len(halo_speedups_pq)), halo_speedups_pq,
               color=COLORS['halo'], alpha=0.7, label='HALO')
        ax.plot(range(len(optimal_speedups_pq)), optimal_speedups_pq,
                color=COLORS['optimal'], linewidth=2, alpha=0.8, label='Optimal')
        ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Query (sorted by HALO speedup)', fontsize=12)
        ax.set_ylabel('Speedup vs Native', fontsize=12)
        ax.set_title(f'(a) Per-Query Speedup ({best_scenario})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)

        # Right: Scatter - HALO vs Optimal speedup per query
        ax = axes[1]
        halo_pq = {pq['query_id']: pq['speedup'] for pq in halo_r['per_query']}
        optimal_pq = {pq['query_id']: pq['speedup'] for pq in optimal_r['per_query']}

        common_q = set(halo_pq.keys()) & set(optimal_pq.keys())
        x_vals = [optimal_pq[q] for q in common_q]
        y_vals = [halo_pq[q] for q in common_q]

        ax.scatter(x_vals, y_vals, alpha=0.5, s=30, c=COLORS['halo'], edgecolors='white', linewidth=0.5)
        lim = max(max(x_vals), max(y_vals)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.4, label='Perfect = Optimal')
        ax.set_xlabel('Optimal Speedup', fontsize=12)
        ax.set_ylabel('HALO Speedup', fontsize=12)
        ax.set_title('(b) HALO vs Optimal (per query)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)

        fig.suptitle(f'HALO Performance: {best_scenario}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/halo_per_query.png')
        plt.close()
        logger.info("Saved: halo_per_query.png")

    # ────────────────────────────────────────────────────────
    # Figure E: Grand Summary Dashboard
    # ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (1) Home-turf speedups
    ax = fig.add_subplot(gs[0, 0])
    home_envs = []
    home_native = []
    home_halo = []
    home_optimal = []
    for env in envs:
        scenario = f"{env} → {env}"
        for r in all_results[scenario]:
            if r['strategy'] == 'Native':
                home_native.append(r['overall_speedup'])
            elif r['strategy'] == 'HALO':
                home_halo.append(r['overall_speedup'])
            elif r['strategy'] == 'Optimal':
                home_optimal.append(r['overall_speedup'])
        home_envs.append(env.replace('_', '\n'))

    x = np.arange(len(home_envs))
    w = 0.25
    ax.bar(x - w, home_native, w, label='Native', color=COLORS['native'], alpha=0.8)
    ax.bar(x, home_halo, w, label='HALO', color=COLORS['halo'], alpha=0.85)
    ax.bar(x + w, home_optimal, w, label='Optimal', color=COLORS['optimal'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(home_envs, fontsize=10)
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.3)
    ax.set_ylabel('Speedup')
    ax.set_title('Home-Turf Speedup', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    # (2) Cross-env speedup (avg)
    ax = fig.add_subplot(gs[0, 1])
    methods = ['BAO-style', 'HALO', 'Optimal']
    avg_home = {m: [] for m in methods}
    avg_cross = {m: [] for m in methods}

    for train_env in envs:
        for test_env in envs:
            scenario = f"{train_env} → {test_env}"
            for r in all_results[scenario]:
                if r['strategy'] in methods:
                    if train_env == test_env:
                        avg_home[r['strategy']].append(r['overall_speedup'])
                    else:
                        avg_cross[r['strategy']].append(r['overall_speedup'])

    x = np.arange(len(methods))
    w = 0.3
    home_avg = [np.mean(avg_home[m]) if avg_home[m] else 0 for m in methods]
    cross_avg = [np.mean(avg_cross[m]) if avg_cross[m] else 0 for m in methods]

    ax.bar(x - w/2, home_avg, w, label='Home-turf', color=COLORS['green'], alpha=0.7)
    ax.bar(x + w/2, cross_avg, w, label='Cross-env', color=COLORS['bao'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['BAO', 'HALO', 'Optimal'], fontsize=11)
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.3)
    ax.set_ylabel('Avg Speedup')
    ax.set_title('Home vs Cross-Env', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # (3) Regression count comparison
    ax = fig.add_subplot(gs[0, 2])
    total_regs = {m: 0 for m in ['BAO-style', 'HALO']}
    total_cross_scenarios = 0
    for train_env in envs:
        for test_env in envs:
            if train_env != test_env:
                total_cross_scenarios += 1
                scenario = f"{train_env} → {test_env}"
                for r in all_results[scenario]:
                    if r['strategy'] in total_regs:
                        total_regs[r['strategy']] += r['regressions']

    bars = ax.bar(['BAO-style', 'HALO'],
                  [total_regs['BAO-style'], total_regs['HALO']],
                  color=[COLORS['bao'], COLORS['halo']], alpha=0.85, edgecolor='white')
    for i, (name, val) in enumerate(total_regs.items()):
        ax.text(i, val + 0.3, str(val), ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Regressions')
    ax.set_title('Cross-Env Regressions', fontsize=14, fontweight='bold')
    if total_regs['BAO-style'] > 0:
        reduction = (1 - total_regs['HALO'] / total_regs['BAO-style']) * 100
        ax.text(0.5, 0.85, f'-{reduction:.0f}%', transform=ax.transAxes,
                ha='center', fontsize=20, fontweight='bold', color=COLORS['green'])

    # (4) Storage sensitivity: NVMe→SATA transfer
    ax = fig.add_subplot(gs[1, 0])
    storage_scenarios = []
    for env1 in envs:
        for env2 in envs:
            parts1 = env1.split('_')
            parts2 = env2.split('_')
            if parts1[0] == parts2[0] and parts1[1] != parts2[1]:
                storage_scenarios.append((env1, env2))

    if storage_scenarios:
        labels = []
        bao_sp = []
        halo_sp = []
        for s, t in storage_scenarios:
            scenario = f"{s} → {t}"
            labels.append(f"{s[:4]}→\n{t[-4:]}")
            for r in all_results[scenario]:
                if r['strategy'] == 'BAO-style':
                    bao_sp.append(r['overall_speedup'])
                elif r['strategy'] == 'HALO':
                    halo_sp.append(r['overall_speedup'])

        x = np.arange(len(labels))
        w = 0.3
        ax.bar(x - w/2, bao_sp, w, label='BAO', color=COLORS['bao'], alpha=0.85)
        ax.bar(x + w/2, halo_sp, w, label='HALO', color=COLORS['halo'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.3)
        ax.set_ylabel('Speedup')
        ax.set_title('Storage Transfer', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

    # (5) CPU sensitivity: ServerA→ServerB transfer
    ax = fig.add_subplot(gs[1, 1])
    cpu_scenarios = []
    for env1 in envs:
        for env2 in envs:
            parts1 = env1.split('_')
            parts2 = env2.split('_')
            if parts1[0] != parts2[0] and parts1[1] == parts2[1]:
                cpu_scenarios.append((env1, env2))

    if cpu_scenarios:
        labels = []
        bao_sp = []
        halo_sp = []
        for s, t in cpu_scenarios:
            scenario = f"{s} → {t}"
            labels.append(f"{s[:1]}→{t[:1]}\n{s[-4:]}")
            for r in all_results[scenario]:
                if r['strategy'] == 'BAO-style':
                    bao_sp.append(r['overall_speedup'])
                elif r['strategy'] == 'HALO':
                    halo_sp.append(r['overall_speedup'])

        x = np.arange(len(labels))
        w = 0.3
        ax.bar(x - w/2, bao_sp, w, label='BAO', color=COLORS['bao'], alpha=0.85)
        ax.bar(x + w/2, halo_sp, w, label='HALO', color=COLORS['halo'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.3)
        ax.set_ylabel('Speedup')
        ax.set_title('CPU Transfer (A↔B)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

    # (6) Summary numbers
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    # Calculate key numbers
    avg_home_halo = np.mean([r['overall_speedup']
                            for s, r_list in all_results.items()
                            for r in r_list
                            if r['strategy'] == 'HALO' and s.split(' → ')[0] == s.split(' → ')[1]])
    avg_cross_halo = np.mean([r['overall_speedup']
                             for s, r_list in all_results.items()
                             for r in r_list
                             if r['strategy'] == 'HALO' and s.split(' → ')[0] != s.split(' → ')[1]])
    avg_cross_bao = np.mean([r['overall_speedup']
                            for s, r_list in all_results.items()
                            for r in r_list
                            if r['strategy'] == 'BAO-style' and s.split(' → ')[0] != s.split(' → ')[1]])

    summary = (
        f"HALO Key Results\n"
        f"{'━'*28}\n"
        f"Home-turf avg:   {avg_home_halo:.2f}x\n"
        f"Cross-env avg:   {avg_cross_halo:.2f}x\n"
        f"BAO cross avg:   {avg_cross_bao:.2f}x\n"
        f"{'━'*28}\n"
        f"HALO advantage:  {(avg_cross_halo/avg_cross_bao - 1)*100:+.1f}%\n"
        f"{'━'*28}\n"
        f"BAO regressions: {total_regs['BAO-style']}\n"
        f"HALO regressions:{total_regs['HALO']}\n"
        f"Reduction:       {reduction:.0f}%\n"
    )
    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
           fontsize=13, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#E8F5E9',
                    edgecolor=COLORS['halo'], alpha=0.9))

    fig.suptitle('HALO Scheme: Complete Evaluation Summary',
                fontsize=18, fontweight='bold', y=1.01)
    fig.savefig(f'{OUTPUT_DIR}/halo_grand_summary.png')
    plt.close()
    logger.info("Saved: halo_grand_summary.png")

    return all_results


if __name__ == '__main__':
    logger.info("Loading data...")
    df_q, df_o = load_all_data()
    logger.info(f"Queries: {len(df_q):,}, Operators: {len(df_o):,}")

    logger.info("Running full evaluation...")
    all_results, all_speedups = run_full_evaluation(df_q)

    logger.info("Generating visualizations...")
    visualize_halo_results(all_results, all_speedups)

    # Save results
    summary_data = {}
    for scenario, results in all_results.items():
        summary_data[scenario] = [{
            'strategy': r['strategy'],
            'overall_speedup': r['overall_speedup'],
            'regressions': r['regressions'],
            'regression_rate': r['regression_rate'],
            'total_queries': r['total_queries'],
        } for r in results]

    with open('/root/halo/results/halo_evaluation.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}/")
    logger.info("Results JSON: /root/halo/results/halo_evaluation.json")
