# -*- coding: utf-8 -*-
"""
Halo Phase 4v2: Improved HALO Sigma Model

Improvements over v1 (WITHOUT requiring repeated runs):
  1. Filter out noise: remove operators with self_time < 0.1ms
  2. One-hot encoding for op_type (instead of label encoding)
  3. Better features: log_est_cost, is_join, is_scan, actual_rows/est_rows ratio
  4. Weighted sampling to balance env pairs
  5. Table-alias-aware matching for cross-env operator pairs
  6. Outlier removal (|delta| > 5 = extreme outliers)
  7. Hyperparameter tuning via grid search
"""

import pandas as pd
import numpy as np
import json
import os
import math
import pickle
import logging
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ENV_MAP = {
    'A_NVMe': {'server': 'A', 'storage': 'NVMe', 'cpu': 'i9-12900K'},
    'A_SATA': {'server': 'A', 'storage': 'SATA', 'cpu': 'i9-12900K'},
    'B_NVMe': {'server': 'B', 'storage': 'NVMe', 'cpu': 'EPYC-7713'},
    'B_SATA': {'server': 'B', 'storage': 'SATA', 'cpu': 'EPYC-7713'},
}

ENV_PAIRS = []
for e1 in sorted(ENV_MAP.keys()):
    for e2 in sorted(ENV_MAP.keys()):
        if e1 < e2:
            ENV_PAIRS.append({
                'env1': e1, 'env2': e2,
                'storage_changed': int(ENV_MAP[e1]['storage'] != ENV_MAP[e2]['storage']),
                'compute_changed': int(ENV_MAP[e1]['cpu'] != ENV_MAP[e2]['cpu']),
                'pair_label': f"{e1}_vs_{e2}"
            })


# ── Improvement 1: Better filtering ──

def filter_noisy_operators(df_ops, min_self_time=0.1, min_actual_rows=1):
    """Remove operators that are too fast to measure reliably."""
    before = len(df_ops)
    df = df_ops[
        (df_ops['self_time'] >= min_self_time) |  # Keep if self_time is measurable
        (df_ops['actual_rows'] >= 100)  # Or if it processes significant rows
    ].copy()
    logger.info(f"  Noise filter: {before:,} -> {len(df):,} "
               f"(removed {before - len(df):,} near-zero operators)")
    return df


# ── Improvement 2: Better features ──

def engineer_features(row1, row2, pair):
    """
    Extract rich feature vector from an operator pair.
    Much more informative than v1's 9 features.
    """
    features = {}

    # Op type one-hot (top categories)
    op_types = ['NL_INNER_JOIN', 'FILTER', 'TABLE_SCAN', 'INDEX_LOOKUP',
                'HASH_JOIN', 'AGGREGATE', 'SORT', 'NL_ANTIJOIN',
                'NL_SEMIJOIN', 'INDEX_SCAN', 'TEMP_TABLE_SCAN',
                'STREAM', 'MATERIALIZE', 'GROUP_AGGREGATE', 'NL_LEFT_JOIN']
    for ot in op_types:
        features[f'is_{ot}'] = 1.0 if row1['op_type'] == ot else 0.0

    # Semantic groups
    features['is_join'] = 1.0 if 'JOIN' in row1['op_type'] else 0.0
    features['is_scan'] = 1.0 if 'SCAN' in row1['op_type'] else 0.0
    features['is_index_op'] = 1.0 if 'INDEX' in row1['op_type'] else 0.0
    features['is_agg_sort'] = 1.0 if row1['op_type'] in ('AGGREGATE', 'SORT', 'GROUP_AGGREGATE') else 0.0

    # Cardinality features (from env1)
    features['log_actual_rows'] = math.log(max(row1['actual_rows'], 1))
    features['log_est_rows'] = math.log(max(row1['est_rows'], 1))
    features['log_loops'] = math.log(max(row1['loops'], 1))
    features['log_est_cost'] = math.log(max(row1['est_cost'], 0.01))

    # Ratio features
    features['row_est_error'] = row1['row_est_error']
    features['rows_per_loop'] = math.log(max(row1['actual_rows'], 1) / max(row1['loops'], 1) + 1)

    # Work intensity
    features['log_self_time'] = math.log(max(row1['self_time'], 0.001))
    features['log_total_work'] = math.log(max(row1['total_work'], 0.001))

    # Structure
    features['depth'] = row1['depth']
    features['num_children'] = row1.get('num_children', 0)

    # Environment factor indicators
    features['storage_changed'] = pair['storage_changed']
    features['compute_changed'] = pair['compute_changed']
    features['both_changed'] = pair['storage_changed'] * pair['compute_changed']

    return features


# ── Improvement 3: Better matching ──

def build_env_pairs_v2(df_ops):
    """
    Build training dataset with improved matching and features.
    Uses (query_id, hint_variant, hint_set, op_type, table_alias) for matching
    when table_alias is available, falling back to tree_position.
    """
    logger.info("Building improved env-pair dataset...")

    # Filter noise first
    df_ops = filter_noisy_operators(df_ops)

    # Create matching keys (prefer table_alias, fallback to position)
    df_ops = df_ops.copy()
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

        # Try alias matching first (more accurate)
        ops1_alias = ops1[ops1['table_alias'] != ''].set_index('match_key_alias')
        ops2_alias = ops2[ops2['table_alias'] != ''].set_index('match_key_alias')
        matched_alias = ops1_alias.index.intersection(ops2_alias.index)

        # Position matching for the rest
        ops1_pos = ops1.set_index('match_key_pos')
        ops2_pos = ops2.set_index('match_key_pos')
        matched_pos = ops1_pos.index.intersection(ops2_pos.index)

        # Combine (alias has priority)
        used_keys = set()
        match_pairs = []

        for key in matched_alias:
            try:
                o1 = ops1_alias.loc[key]
                o2 = ops2_alias.loc[key]
                if isinstance(o1, pd.DataFrame): o1 = o1.iloc[0]
                if isinstance(o2, pd.DataFrame): o2 = o2.iloc[0]
                if o1['op_type'] == o2['op_type']:
                    match_pairs.append((o1, o2, 'alias'))
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
                    match_pairs.append((o1, o2, 'position'))
            except Exception:
                continue

        pair_count = 0
        for o1, o2, match_type in match_pairs:
            st1 = max(o1['self_time'], 0.001)
            st2 = max(o2['self_time'], 0.001)
            delta = math.log(st2 / st1)

            # Improvement 4: Outlier removal
            if abs(delta) > 5:  # > ~150x difference is likely data error
                continue

            feat = engineer_features(o1, o2, pair)
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
            feat['match_type'] = match_type

            samples.append(feat)
            pair_count += 1

        logger.info(f"  {e1} vs {e2}: {pair_count:,} matched pairs")

    df_samples = pd.DataFrame(samples)
    logger.info(f"Total training samples: {len(df_samples):,}")
    return df_samples


def train_sigma_v2(df_samples, output_dir='/root/halo/results'):
    """Train improved sigma model with hyperparameter tuning."""
    import xgboost as xgb

    os.makedirs(output_dir, exist_ok=True)

    # Feature columns (all except meta columns)
    meta_cols = ['delta_time', 'self_time_env1', 'self_time_env2',
                 'query_id', 'benchmark', 'hint_set', 'op_type',
                 'env1', 'env2', 'pair_label', 'match_type']
    feature_cols = [c for c in df_samples.columns if c not in meta_cols]

    X = df_samples[feature_cols].values.astype(np.float32)
    y = df_samples['delta_time'].values.astype(np.float32)

    # Clean
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    df_clean = df_samples[mask].reset_index(drop=True)

    logger.info(f"\nTraining improved sigma model...")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Target stats: mean={y.mean():.4f}, std={y.std():.4f}")

    # ── Improvement 5: Weighted samples for env pair balance ──
    pair_counts = df_clean['pair_label'].value_counts()
    max_count = pair_counts.max()
    weights = df_clean['pair_label'].map(lambda p: max_count / pair_counts.get(p, 1)).values

    # ── Improvement 6: Hyperparameter tuning ──
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'min_child_weight': [5, 10],
    }

    # Quick grid search on a subset
    n_search = min(10000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_search, replace=False)

    best_params = {
        'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.05, 'subsample': 0.8,
        'min_child_weight': 5, 'tree_method': 'hist',
        'random_state': 42,
    }

    # Try a few configs
    best_score = -np.inf
    for depth in [4, 6, 8]:
        for lr in [0.05, 0.1]:
            for mcw in [5, 10]:
                m = xgb.XGBRegressor(
                    n_estimators=300, max_depth=depth,
                    learning_rate=lr, subsample=0.8,
                    min_child_weight=mcw, tree_method='hist',
                    random_state=42)
                m.fit(X[idx[:7000]], y[idx[:7000]],
                      sample_weight=weights[idx[:7000]])
                score = r2_score(y[idx[7000:]], m.predict(X[idx[7000:]]))
                if score > best_score:
                    best_score = score
                    best_params.update({'max_depth': depth, 'learning_rate': lr,
                                       'min_child_weight': mcw})

    logger.info(f"  Best params: depth={best_params['max_depth']}, "
               f"lr={best_params['learning_rate']}, mcw={best_params['min_child_weight']}")
    logger.info(f"  Validation R2: {best_score:.4f}")

    # ── Leave-1-env-pair-out CV with best params ──
    model = xgb.XGBRegressor(**best_params)
    groups = df_clean['pair_label'].values
    logo = LeaveOneGroupOut()

    y_pred_cv = np.zeros_like(y)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        model.fit(X[train_idx], y[train_idx],
                  sample_weight=weights[train_idx])
        y_pred = model.predict(X[test_idx])
        y_pred_cv[test_idx] = y_pred

        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred) if len(test_idx) > 1 else 0
        pair_name = groups[test_idx[0]]

        # Direction accuracy (is the sign of delta correct?)
        nonzero = np.abs(y[test_idx]) > 0.1
        if nonzero.sum() > 0:
            dir_acc = ((y[test_idx][nonzero] > 0) ==
                       (y_pred[nonzero] > 0)).mean() * 100
        else:
            dir_acc = 100.0

        fold_results.append({
            'pair': pair_name, 'n': len(test_idx),
            'rmse': float(rmse), 'r2': float(r2), 'dir_acc': float(dir_acc)
        })
        logger.info(f"  Fold {fold_idx+1} ({pair_name}): RMSE={rmse:.3f}, "
                    f"R2={r2:.3f}, DirAcc={dir_acc:.1f}%, n={len(test_idx)}")

    # Overall metrics
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    overall_r2 = r2_score(y, y_pred_cv)
    overall_corr = np.corrcoef(y, y_pred_cv)[0, 1]

    nonzero_mask = np.abs(y) > 0.1
    if nonzero_mask.sum() > 0:
        overall_dir_acc = ((y[nonzero_mask] > 0) ==
                           (y_pred_cv[nonzero_mask] > 0)).mean() * 100
    else:
        overall_dir_acc = 100.0

    # Ranking accuracy: for pairs of operators, is the order preserved?
    n_pairs_check = min(50000, len(y) * (len(y) - 1) // 2)
    rng = np.random.RandomState(42)
    rank_correct = 0
    rank_total = 0
    for _ in range(n_pairs_check):
        i, j = rng.choice(len(y), 2, replace=False)
        if abs(y[i] - y[j]) > 0.1:
            rank_total += 1
            if (y[i] > y[j]) == (y_pred_cv[i] > y_pred_cv[j]):
                rank_correct += 1
    ranking_acc = (rank_correct / max(rank_total, 1)) * 100

    # High-Risk Classification Metrics (|delta| > 0.5)
    y_high = np.abs(y) > 0.5
    y_pred_high = np.abs(y_pred_cv) > 0.5
    high_prec = precision_score(y_high, y_pred_high) if y_pred_high.any() else 0.0
    high_rec = recall_score(y_high, y_pred_high) if y_high.any() else 0.0
    high_f1 = f1_score(y_high, y_pred_high) if y_high.any() else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"IMPROVED Sigma Model Results:")
    logger.info(f"  RMSE:             {overall_rmse:.3f}")
    logger.info(f"  R2:               {overall_r2:.3f}")
    logger.info(f"  Pearson r:        {overall_corr:.3f}")
    logger.info(f"  Direction Acc:    {overall_dir_acc:.1f}%")
    logger.info(f"  Ranking Acc:      {ranking_acc:.1f}%")
    logger.info(f"  High-Risk (abs>0.5) Detection:")
    logger.info(f"    Precision:      {high_prec:.3f}")
    logger.info(f"    Recall:         {high_rec:.3f}")
    logger.info(f"    F1-Score:       {high_f1:.3f}")

    # Train final model on all data
    model.fit(X, y, sample_weight=weights)

    # Feature importance (top 15)
    imp = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
    logger.info(f"\nTop 15 Feature Importance:")
    for fname, importance in feat_imp[:15]:
        bar = '#' * int(importance * 50)
        logger.info(f"  {fname:<25} {importance:.3f} {bar}")

    # ── Operator sigma analysis (improved) ──
    df_clean['predicted_delta'] = y_pred_cv
    op_stats = []
    logger.info(f"\nOperator Sigma (v2):")
    logger.info(f"{'Op Type':<22} {'N':>6} {'sigma_S':>8} {'sigma_C':>8} {'R2':>6} {'DirAcc':>7}")
    logger.info("-" * 60)

    for op_type, group in df_clean.groupby('op_type'):
        if len(group) < 20:
            continue
        sg = group[group['storage_changed'] == 1]
        cg = group[group['compute_changed'] == 1]
        sig_s = sg['delta_time'].abs().mean() if len(sg) > 0 else 0
        sig_c = cg['delta_time'].abs().mean() if len(cg) > 0 else 0

        # Per-operator accuracy
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

    # Save results
    results = {
        'model_version': 'v2_improved',
        'improvements': [
            'noise_filtering', 'one_hot_encoding', 'rich_features',
            'weighted_sampling', 'alias_matching', 'outlier_removal',
            'hyperparameter_tuning'
        ],
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'overall_rmse': float(overall_rmse),
        'overall_r2': float(overall_r2),
        'pearson_r': float(overall_corr),
        'direction_accuracy': float(overall_dir_acc),
        'ranking_accuracy': float(ranking_acc),
        'high_risk_metrics': {
            'precision': float(high_prec),
            'recall': float(high_rec),
            'f1': float(high_f1)
        },
        'fold_results': fold_results,
        'feature_importance': {k: float(v) for k, v in feat_imp[:20]},
        'operator_sigma': op_stats,
        'best_params': {k: v for k, v in best_params.items() if k != 'random_state'},
    }

    with open(os.path.join(output_dir, 'sigma_results_v2.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(output_dir, 'sigma_model_v2.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)

    df_clean.to_parquet(os.path.join(output_dir, 'sigma_predictions_v2.parquet'), index=False)

    # Comparison with v1
    v1_path = os.path.join(output_dir, 'sigma_results.json')
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        logger.info(f"\n{'='*60}")
        logger.info(f"V1 vs V2 Comparison:")
        logger.info(f"  {'Metric':<20} {'V1':>12} {'V2':>12} {'Change':>12}")
        logger.info(f"  {'-'*56}")
        logger.info(f"  {'Pearson r':<20} {v1.get('pearson_r',0):>12.3f} {overall_corr:>12.3f} {overall_corr - v1.get('pearson_r',0):>+12.3f}")
        logger.info(f"  {'Ranking Acc':<20} {v1.get('ranking_accuracy',0):>11.1f}% {ranking_acc:>11.1f}% {ranking_acc - v1.get('ranking_accuracy',0):>+11.1f}%")
        logger.info(f"  {'RMSE':<20} {v1.get('overall_rmse',0):>12.3f} {overall_rmse:>12.3f} {overall_rmse - v1.get('overall_rmse',0):>+12.3f}")
        logger.info(f"  {'Direction Acc':<20} {'N/A':>12} {overall_dir_acc:>11.1f}%")
        logger.info(f"  {'R2':<20} {'N/A':>12} {overall_r2:>12.3f}")

    return model, results


if __name__ == '__main__':
    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    logger.info(f"Loaded {len(df_ops):,} operators")
    df_samples = build_env_pairs_v2(df_ops)
    df_samples.to_parquet('/root/halo/results/sigma_training_v2.parquet', index=False)
    model, results = train_sigma_v2(df_samples)
