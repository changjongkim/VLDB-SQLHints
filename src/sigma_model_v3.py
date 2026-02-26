# -*- coding: utf-8 -*-
"""
Halo Phase 4v3: Enhanced HALO Sigma Model

Key improvements over v2:
  1. Interaction features: op_type × hw_change crossterms
  2. Ratio features: cost_per_row, time_per_row, selectivity
  3. Relative magnitude features capturing operator "weight" in query
  4. Stacked ensemble: XGBoost + LightGBM averaged
  5. Proper visualization with REAL CV predictions (not mock data)
  6. Comprehensive evaluation plots for paper
"""

import pandas as pd
import numpy as np
import json
import os
import math
import pickle
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             precision_score, recall_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from hw_registry import HW_REGISTRY, compute_hw_features, get_env_pairs
from workload_metadata import enrich_operator

# Backward compatibility aliases
ENV_MAP = HW_REGISTRY
ENV_PAIRS = get_env_pairs(include_target_envs=False)


def filter_noisy_operators(df_ops, min_self_time=0.1):
    """Remove operators that are too fast to measure reliably."""
    before = len(df_ops)
    df = df_ops[
        (df_ops['self_time'] >= min_self_time) |
        (df_ops['actual_rows'] >= 100)
    ].copy()
    logger.info(f"  Noise filter: {before:,} -> {len(df):,} "
               f"(removed {before - len(df):,} near-zero operators)")
    return df


def engineer_features_v3(row1, row2, pair):
    """
    Enhanced feature engineering with:
      - Operator type one-hot + semantic groups
      - Cardinality, estimation quality, work intensity
      - Ratio features (cost_per_row, time_per_row, selectivity)
      - Binary HW change flags (backward compatible)
      - Quantitative HW ratios (storage speed, IOPS, CPU clock/cores, RAM)
      - Workload metadata (table size, row count, indexes, buffer fit)
      - Operator × HW interaction terms (binary AND quantitative)
      - Workload × HW cross-terms
    """
    features = {}

    # ── 1. Op type one-hot (15 types) ──
    op_types = ['NL_INNER_JOIN', 'FILTER', 'TABLE_SCAN', 'INDEX_LOOKUP',
                'HASH_JOIN', 'AGGREGATE', 'SORT', 'NL_ANTIJOIN',
                'NL_SEMIJOIN', 'INDEX_SCAN', 'TEMP_TABLE_SCAN',
                'STREAM', 'MATERIALIZE', 'GROUP_AGGREGATE', 'NL_LEFT_JOIN']
    for ot in op_types:
        features[f'is_{ot}'] = 1.0 if row1['op_type'] == ot else 0.0

    # ── 2. Semantic groups ──
    features['is_join'] = 1.0 if 'JOIN' in row1['op_type'] else 0.0
    features['is_scan'] = 1.0 if 'SCAN' in row1['op_type'] else 0.0
    features['is_index_op'] = 1.0 if 'INDEX' in row1['op_type'] else 0.0
    features['is_agg_sort'] = 1.0 if row1['op_type'] in ('AGGREGATE', 'SORT', 'GROUP_AGGREGATE') else 0.0

    # ── 3. Cardinality features ──
    actual_rows = max(row1['actual_rows'], 1)
    est_rows = max(row1['est_rows'], 1)
    loops = max(row1['loops'], 1)
    est_cost = max(row1['est_cost'], 0.01)
    self_time = max(row1['self_time'], 0.001)
    total_work = max(row1['total_work'], 0.001)

    features['log_actual_rows'] = math.log(actual_rows)
    features['log_est_rows'] = math.log(est_rows)
    features['log_loops'] = math.log(loops)
    features['log_est_cost'] = math.log(est_cost)

    # ── 4. Estimation quality ──
    features['row_est_error'] = row1['row_est_error']
    features['rows_per_loop'] = math.log(actual_rows / loops + 1)

    # ── 5. Work intensity ──
    features['log_self_time'] = math.log(self_time)
    features['log_total_work'] = math.log(total_work)

    # ── 6. Ratio features ──
    features['cost_per_row'] = math.log(est_cost / est_rows + 0.001)
    features['time_per_row'] = math.log(self_time / actual_rows + 0.0001)
    features['selectivity'] = math.log(actual_rows / est_rows + 0.001)

    # ── 7. Structure ──
    features['depth'] = row1['depth']
    features['num_children'] = row1.get('num_children', 0)

    # ── 8. Binary HW change flags (backward compatible) ──
    sc = pair['storage_changed']
    cc = pair['compute_changed']
    features['storage_changed'] = sc
    features['compute_changed'] = cc
    features['both_changed'] = pair.get('both_changed', sc * cc)

    # ── 9. Quantitative HW ratios (positive = target is better) ──
    features['storage_speed_ratio'] = pair.get('storage_speed_ratio', 0.0)
    features['iops_ratio'] = pair.get('iops_ratio', 0.0)
    features['write_speed_ratio'] = pair.get('write_speed_ratio', 0.0)
    features['cpu_core_ratio'] = pair.get('cpu_core_ratio', 0.0)
    features['cpu_thread_ratio'] = pair.get('cpu_thread_ratio', 0.0)
    features['cpu_clock_ratio'] = pair.get('cpu_clock_ratio', 0.0)
    features['cpu_base_clock_ratio'] = pair.get('cpu_base_clock_ratio', 0.0)
    features['l3_cache_ratio'] = pair.get('l3_cache_ratio', 0.0)
    features['ram_ratio'] = pair.get('ram_ratio', 0.0)
    features['buffer_pool_ratio'] = pair.get('buffer_pool_ratio', 0.0)
    features['is_storage_downgrade'] = pair.get('is_storage_downgrade', 0)
    features['is_cpu_downgrade'] = pair.get('is_cpu_downgrade', 0)
    features['is_ram_downgrade'] = pair.get('is_ram_downgrade', 0)

    # ── 10. Workload metadata features ──
    features['log_table_rows'] = math.log(max(row1.get('table_rows', 1), 1))
    features['log_table_size_mb'] = math.log(max(row1.get('table_size_mb', 1), 1))
    features['n_indexes'] = row1.get('n_indexes', 0)
    features['log_dataset_gb'] = math.log(max(row1.get('dataset_total_gb', 0.1), 0.1))
    features['fits_in_buffer'] = row1.get('fits_in_buffer', 0)

    # ── 11. Operator × Binary HW interaction ──
    features['scan_x_storage'] = features['is_scan'] * sc
    features['scan_x_compute'] = features['is_scan'] * cc
    features['join_x_storage'] = features['is_join'] * sc
    features['join_x_compute'] = features['is_join'] * cc
    features['index_x_storage'] = features['is_index_op'] * sc
    features['index_x_compute'] = features['is_index_op'] * cc
    features['agg_x_storage'] = features['is_agg_sort'] * sc
    features['agg_x_compute'] = features['is_agg_sort'] * cc

    # ── 12. Work × Binary HW interaction ──
    features['work_x_storage'] = features['log_total_work'] * sc
    features['work_x_compute'] = features['log_total_work'] * cc
    features['rows_x_storage'] = features['log_actual_rows'] * sc
    features['rows_x_compute'] = features['log_actual_rows'] * cc

    # ── 13. Operator × Quantitative HW interaction ──
    #   "TABLE_SCAN on 5000x slower storage" vs "TABLE_SCAN on 2x slower storage"
    iops_r = pair.get('iops_ratio', 0.0)
    clock_r = pair.get('cpu_clock_ratio', 0.0)
    core_r = pair.get('cpu_core_ratio', 0.0)
    features['scan_x_iops'] = features['is_scan'] * iops_r
    features['scan_x_storage_speed'] = features['is_scan'] * pair.get('storage_speed_ratio', 0.0)
    features['join_x_cpu_clock'] = features['is_join'] * clock_r
    features['join_x_cpu_cores'] = features['is_join'] * core_r
    features['agg_x_cpu_clock'] = features['is_agg_sort'] * clock_r
    features['index_x_iops'] = features['is_index_op'] * iops_r

    # ── 14. Work × Quantitative HW interaction ──
    features['rows_x_iops'] = features['log_actual_rows'] * iops_r
    features['work_x_cpu_clock'] = features['log_total_work'] * clock_r
    features['cost_x_iops'] = features['log_est_cost'] * iops_r

    # ── 15. Workload × HW cross-terms ──
    #   "22GB table + IOPS downgrade" = very dangerous
    features['table_size_x_iops'] = features['log_table_size_mb'] * iops_r
    features['table_size_x_storage_speed'] = features['log_table_size_mb'] * pair.get('storage_speed_ratio', 0.0)
    features['table_rows_x_iops'] = features['log_table_rows'] * iops_r
    features['buffer_miss_x_iops'] = (1 - features['fits_in_buffer']) * iops_r
    features['dataset_x_ram'] = features['log_dataset_gb'] * pair.get('ram_ratio', 0.0)

    return features


def build_env_pairs_v3(df_ops):
    """Build training dataset with v3 features."""
    logger.info("Building v3 env-pair dataset...")
    df_ops = filter_noisy_operators(df_ops)

    df_ops = df_ops.copy()

    # ── Enrich operators with workload metadata ──
    logger.info("  Enriching operators with workload metadata...")
    for idx, row in df_ops.iterrows():
        enriched = enrich_operator(
            row.to_dict(),
            benchmark=row.get('benchmark', ''),
            env_id=row.get('env_id', None)
        )
        for col in ['table_rows', 'table_size_mb', 'n_indexes',
                     'dataset_total_gb', 'n_tables_in_dataset', 'fits_in_buffer']:
            df_ops.at[idx, col] = enriched[col]
    logger.info(f"  Enrichment complete: {len(df_ops):,} operators")

    df_ops['match_key_alias'] = (
        df_ops['query_id'] + '|' +
        df_ops['hint_variant'] + '|' +
        df_ops['hint_set'] + '|' +
        df_ops['op_type'] + '|' +
        df_ops['table_alias'] + '|' +
        df_ops['benchmark']
    )
    df_ops['match_key_pos'] = (
        df_ops['query_id'] + '|' +
        df_ops['hint_variant'] + '|' +
        df_ops['hint_set'] + '|' +
        df_ops['tree_position'].astype(str) + '|' +
        df_ops['benchmark']
    )

    samples = []
    for pair in ENV_PAIRS:
        e1, e2 = pair['env1'], pair['env2']
        ops1 = df_ops[df_ops['env_id'] == e1]
        ops2 = df_ops[df_ops['env_id'] == e2]
        if len(ops1) == 0 or len(ops2) == 0:
            continue

        ops1_alias = ops1[ops1['table_alias'] != ''].set_index('match_key_alias')
        ops2_alias = ops2[ops2['table_alias'] != ''].set_index('match_key_alias')
        matched_alias = ops1_alias.index.intersection(ops2_alias.index)

        ops1_pos = ops1.set_index('match_key_pos')
        ops2_pos = ops2.set_index('match_key_pos')
        matched_pos = ops1_pos.index.intersection(ops2_pos.index)

        used_keys = set()
        match_pairs = []

        for key in matched_alias:
            try:
                o1 = ops1_alias.loc[key]
                o2 = ops2_alias.loc[key]
                if isinstance(o1, pd.DataFrame): o1 = o1.iloc[0]
                if isinstance(o2, pd.DataFrame): o2 = o2.iloc[0]
                if o1['op_type'] == o2['op_type']:
                    match_pairs.append((o1, o2))
                    used_keys.add((o1['query_id'], o1['tree_position']))
            except Exception:
                continue

        for key in matched_pos:
            try:
                o1 = ops1_pos.loc[key]
                o2 = ops2_pos.loc[key]
                if isinstance(o1, pd.DataFrame): o1 = o1.iloc[0]
                if isinstance(o2, pd.DataFrame): o2 = o2.iloc[0]
                if (o1['query_id'], o1['tree_position']) in used_keys:
                    continue
                if o1['op_type'] == o2['op_type']:
                    match_pairs.append((o1, o2))
            except Exception:
                continue

        pair_count = 0
        for o1, o2 in match_pairs:
            st1 = max(o1['self_time'], 0.001)
            st2 = max(o2['self_time'], 0.001)
            delta = math.log(st2 / st1)
            if abs(delta) > 5:
                continue

            feat = engineer_features_v3(o1, o2, pair)
            feat['delta_time'] = delta
            feat['self_time_env1'] = st1
            feat['self_time_env2'] = st2
            feat['query_id'] = o1['query_id']
            feat['benchmark'] = o1['benchmark']
            feat['hint_set'] = o1['hint_set']
            feat['op_type'] = o1['op_type']
            feat['env1'] = e1
            feat['env2'] = e2
            feat['pair_label'] = pair['pair_label']
            samples.append(feat)
            pair_count += 1

        logger.info(f"  {e1} vs {e2}: {pair_count:,} matched pairs")

    df_samples = pd.DataFrame(samples)
    logger.info(f"Total training samples: {len(df_samples):,}")
    return df_samples


def train_sigma_v3(df_samples, output_dir='/root/halo/results'):
    """Train v3 sigma model using Random Forest + comprehensive evaluation."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    meta_cols = ['delta_time', 'self_time_env1', 'self_time_env2',
                 'query_id', 'benchmark', 'hint_set', 'op_type',
                 'env1', 'env2', 'pair_label']
    feature_cols = [c for c in df_samples.columns if c not in meta_cols]

    X = df_samples[feature_cols].values.astype(np.float32)
    y = df_samples['delta_time'].values.astype(np.float32)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    df_clean = df_samples[mask].reset_index(drop=True)

    logger.info(f"\nTraining v3 sigma model (Random Forest)...")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Target stats: mean={y.mean():.4f}, std={y.std():.4f}")

    # Weighted samples for env pair balance
    pair_counts = df_clean['pair_label'].value_counts()
    max_count = pair_counts.max()
    weights = df_clean['pair_label'].map(lambda p: max_count / pair_counts.get(p, 1)).values

    # ── Quick HP search for RF ──
    best_params = None
    best_score = -np.inf
    n_search = min(10000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_search, replace=False)

    logger.info("  Hyperparameter search...")
    for n_est in [200, 500]:
        for depth in [8, 12, 16, None]:
            for msl in [2, 5, 10]:
                m = RandomForestRegressor(
                    n_estimators=n_est, max_depth=depth,
                    min_samples_leaf=msl, max_features='sqrt',
                    n_jobs=-1, random_state=42)
                m.fit(X[idx[:7000]], y[idx[:7000]],
                      sample_weight=weights[idx[:7000]])
                score = r2_score(y[idx[7000:]], m.predict(X[idx[7000:]]))
                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_estimators': n_est, 'max_depth': depth,
                        'min_samples_leaf': msl, 'max_features': 'sqrt',
                        'n_jobs': -1, 'random_state': 42
                    }

    logger.info(f"  Best RF params: n_est={best_params['n_estimators']}, "
               f"depth={best_params['max_depth']}, msl={best_params['min_samples_leaf']}")
    logger.info(f"  Validation R2: {best_score:.4f}")

    # ── Leave-1-env-pair-out CV ──
    groups = df_clean['pair_label'].values
    logo = LeaveOneGroupOut()

    y_pred_cv = np.zeros_like(y)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        rf = RandomForestRegressor(**best_params)
        rf.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_pred_cv[test_idx] = y_pred

        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred) if len(test_idx) > 1 else 0
        pair_name = groups[test_idx[0]]

        nonzero = np.abs(y[test_idx]) > 0.1
        dir_acc = ((y[test_idx][nonzero] > 0) == (y_pred[nonzero] > 0)).mean() * 100 if nonzero.sum() > 0 else 100.0

        fold_results.append({
            'pair': pair_name, 'n': len(test_idx),
            'rmse': float(rmse), 'r2': float(r2), 'dir_acc': float(dir_acc)
        })
        logger.info(f"  Fold {fold_idx+1} ({pair_name}): RMSE={rmse:.3f}, "
                    f"R2={r2:.3f}, DirAcc={dir_acc:.1f}%, n={len(test_idx)}")

    # ── Overall metrics ──
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    overall_r2 = r2_score(y, y_pred_cv)
    overall_mae = mean_absolute_error(y, y_pred_cv)
    overall_corr = np.corrcoef(y, y_pred_cv)[0, 1]

    nonzero_mask = np.abs(y) > 0.1
    overall_dir_acc = ((y[nonzero_mask] > 0) == (y_pred_cv[nonzero_mask] > 0)).mean() * 100 if nonzero_mask.sum() > 0 else 100.0

    # Ranking accuracy
    n_pairs_check = min(50000, len(y) * (len(y) - 1) // 2)
    rng = np.random.RandomState(42)
    rank_correct = rank_total = 0
    for _ in range(n_pairs_check):
        i, j = rng.choice(len(y), 2, replace=False)
        if abs(y[i] - y[j]) > 0.1:
            rank_total += 1
            if (y[i] > y[j]) == (y_pred_cv[i] > y_pred_cv[j]):
                rank_correct += 1
    ranking_acc = (rank_correct / max(rank_total, 1)) * 100

    # High-Risk Classification
    for threshold, label in [(0.5, '0.5'), (0.8, '0.8')]:
        y_high = np.abs(y) > threshold
        y_pred_high = np.abs(y_pred_cv) > threshold
        prec = precision_score(y_high, y_pred_high) if y_pred_high.any() else 0.0
        rec = recall_score(y_high, y_pred_high) if y_high.any() else 0.0
        f1 = f1_score(y_high, y_pred_high) if y_high.any() else 0.0
        logger.info(f"  High-Risk (|δ|>{label}): Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

    y_high_final = np.abs(y) > 0.8
    y_pred_high_final = np.abs(y_pred_cv) > 0.8
    high_prec = precision_score(y_high_final, y_pred_high_final) if y_pred_high_final.any() else 0.0
    high_rec = recall_score(y_high_final, y_pred_high_final) if y_high_final.any() else 0.0
    high_f1 = f1_score(y_high_final, y_pred_high_final) if y_high_final.any() else 0.0

    logger.info(f"\n{'='*60}")
    logger.info("σ Model v3 Results (Random Forest):")
    logger.info(f"  RMSE:             {overall_rmse:.3f}")
    logger.info(f"  MAE:              {overall_mae:.3f}")
    logger.info(f"  R²:               {overall_r2:.3f}")
    logger.info(f"  Pearson r:        {overall_corr:.3f}")
    logger.info(f"  Direction Acc:    {overall_dir_acc:.1f}%")
    logger.info(f"  Ranking Acc:      {ranking_acc:.1f}%")
    logger.info(f"  High-Risk Detection (|δ|>0.8):")
    logger.info(f"    Precision:      {high_prec:.3f}")
    logger.info(f"    Recall:         {high_rec:.3f}")
    logger.info(f"    F1-Score:       {high_f1:.3f}")

    # ══════════════════════════════════════════════════════════
    #   VISUALIZATION (Using REAL CV predictions)
    # ══════════════════════════════════════════════════════════

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Predicted vs Actual (Hexbin)
    ax = axes[0, 0]
    hb = ax.hexbin(y, y_pred_cv, gridsize=40, cmap='YlGnBu', mincnt=1)
    ax.plot([-4, 4], [-4, 4], 'r--', linewidth=1.5, label='Perfect (y=x)')
    ax.set_xlabel('Actual Δ (log-ratio)', fontsize=13)
    ax.set_ylabel('Predicted Δ (log-ratio)', fontsize=13)
    ax.set_title(f'(a) Predicted vs Actual Operator Delta\n'
                 f'R²={overall_r2:.3f}, Pearson r={overall_corr:.3f}', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    plt.colorbar(hb, ax=ax, label='Count')

    # Plot 2: Residual Distribution
    ax = axes[0, 1]
    residuals = y - y_pred_cv
    ax.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title(f'(b) Prediction Error Distribution\n'
                 f'MAE={overall_mae:.3f}, Std={residuals.std():.3f}', fontsize=14)

    # Plot 3: Confusion Matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y_high_final, y_pred_high_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=['Safe (|δ|≤0.8)', 'High Risk (|δ|>0.8)'],
                yticklabels=['Safe (|δ|≤0.8)', 'High Risk (|δ|>0.8)'],
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('Actual Label', fontsize=13)
    ax.set_title(f'(c) High-Risk Operator Detection\n'
                 f'Precision={high_prec:.2f}, Recall={high_rec:.2f}, F1={high_f1:.2f}', fontsize=14)

    # Plot 4: Per-Fold R²
    ax = axes[1, 1]
    fold_df = pd.DataFrame(fold_results)
    colors = []
    for _, row in fold_df.iterrows():
        pair = row['pair']
        p = [p for p in ENV_PAIRS if p['pair_label'] == pair][0]
        if p['storage_changed'] and p['compute_changed']:
            colors.append('#e74c3c')
        elif p['storage_changed']:
            colors.append('#3498db')
        elif p['compute_changed']:
            colors.append('#2ecc71')
        else:
            colors.append('#95a5a6')

    ax.barh(fold_df['pair'].str.replace('_vs_', ' → '), fold_df['r2'], color=colors)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('R² Score', fontsize=13)
    ax.set_title(f'(d) Per-Environment-Pair R²\n'
                 f'Overall R²={overall_r2:.3f}', fontsize=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#3498db', label='Storage Only'),
        Patch(facecolor='#2ecc71', label='CPU Only'),
        Patch(facecolor='#e74c3c', label='Both Changed'),
    ], loc='lower right', fontsize=10)

    plt.tight_layout(pad=2.0)
    fig_path = os.path.join(output_dir, 'figures', 'sigma_model_v3_evaluation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nSaved evaluation plot: {fig_path}")
    plt.close()

    # ── Feature Importance Plot ──
    model = RandomForestRegressor(**best_params)
    model.fit(X, y, sample_weight=weights)
    imp = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    top_n = 20
    names = [f[0] for f in feat_imp[:top_n]][::-1]
    values = [f[1] for f in feat_imp[:top_n]][::-1]
    colors2 = ['#e74c3c' if 'x_' in n else '#3498db' for n in names]
    ax2.barh(names, values, color=colors2)
    ax2.set_xlabel('Feature Importance (MDI)', fontsize=13)
    ax2.set_title(f'Top {top_n} Feature Importance (Random Forest)\n'
                  f'(Red = Interaction Features, Blue = Base Features)', fontsize=14)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'figures', 'sigma_feature_importance_v3.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance: {fig2_path}")
    plt.close()

    logger.info(f"\nTop 15 Feature Importance:")
    for fname, importance in feat_imp[:15]:
        bar = '#' * int(importance * 50)
        logger.info(f"  {fname:<30} {importance:.3f} {bar}")

    # ── Operator-level sigma analysis ──
    df_clean['predicted_delta'] = y_pred_cv
    op_stats = []
    logger.info(f"\nOperator Sigma (v3 RF):")
    logger.info(f"{'Op Type':<22} {'N':>6} {'σ_S':>8} {'σ_C':>8} {'R2':>6} {'DirAcc':>7}")
    logger.info("-" * 60)

    for op_type, group in df_clean.groupby('op_type'):
        if len(group) < 20:
            continue
        sg = group[group['storage_changed'] == 1]
        cg = group[group['compute_changed'] == 1]
        sig_s = sg['delta_time'].abs().mean() if len(sg) > 0 else 0
        sig_c = cg['delta_time'].abs().mean() if len(cg) > 0 else 0

        nz = group[group['delta_time'].abs() > 0.1]
        if len(nz) > 1:
            op_dir = ((nz['delta_time'] > 0) == (nz['predicted_delta'] > 0)).mean() * 100
            op_r2 = r2_score(group['delta_time'], group['predicted_delta'])
        else:
            op_dir = 100.0
            op_r2 = 0.0

        logger.info(f"{op_type:<22} {len(group):>6,} {sig_s:>8.3f} {sig_c:>8.3f} "
                    f"{op_r2:>6.3f} {op_dir:>6.1f}%")
        op_stats.append({
            'op_type': op_type, 'count': len(group),
            'sigma_storage': float(sig_s), 'sigma_compute': float(sig_c),
            'r2': float(op_r2), 'dir_acc': float(op_dir),
        })

    # ── Save results ──
    results = {
        'model_type': 'RandomForest',
        'model_version': 'v3_rf',
        'improvements': [
            'random_forest', 'interaction_features', 'ratio_features',
            'real_cv_visualization', 'weighted_sampling'
        ],
        'n_samples': int(len(X)),
        'n_features': len(feature_cols),
        'feature_list': feature_cols,
        'overall_rmse': float(overall_rmse),
        'overall_mae': float(overall_mae),
        'overall_r2': float(overall_r2),
        'pearson_r': float(overall_corr),
        'direction_accuracy': float(overall_dir_acc),
        'ranking_accuracy': float(ranking_acc),
        'high_risk_metrics_0.8': {
            'precision': float(high_prec),
            'recall': float(high_rec),
            'f1': float(high_f1)
        },
        'fold_results': fold_results,
        'feature_importance': {k: float(v) for k, v in feat_imp[:25]},
        'operator_sigma': op_stats,
        'best_params': {k: str(v) for k, v in best_params.items() if k != 'random_state'},
    }

    with open(os.path.join(output_dir, 'sigma_results_v3.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(output_dir, 'sigma_model_v3.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)

    df_clean.to_parquet(os.path.join(output_dir, 'sigma_predictions_v3.parquet'), index=False)

    # ── V2 vs V3 Comparison ──
    v2_path = os.path.join(output_dir, 'sigma_results_v2.json')
    if os.path.exists(v2_path):
        with open(v2_path) as f:
            v2 = json.load(f)
        logger.info(f"\n{'='*60}")
        logger.info("V2 (XGBoost) vs V3 (RandomForest) Comparison:")
        logger.info(f"  {'Metric':<20} {'V2-XGB':>12} {'V3-RF':>12} {'Change':>12}")
        logger.info(f"  {'-'*56}")
        for m_name, v2_key in [('R²', 'overall_r2'), ('Pearson r', 'pearson_r'),
                               ('RMSE', 'overall_rmse'), ('Direction Acc', 'direction_accuracy'),
                               ('Ranking Acc', 'ranking_accuracy')]:
            v2_val = v2.get(v2_key, 0)
            v3_val = results[v2_key]
            change = v3_val - v2_val
            logger.info(f"  {m_name:<20} {v2_val:>12.3f} {v3_val:>12.3f} {change:>+12.3f}")

    return model, results


if __name__ == '__main__':
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    logger.info(f"Loaded {len(df_ops):,} operators")
    df_samples = build_env_pairs_v3(df_ops)
    df_samples.to_parquet('/root/halo/results/sigma_training_v3.parquet', index=False)
    model, results = train_sigma_v3(df_samples)
