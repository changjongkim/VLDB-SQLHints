# -*- coding: utf-8 -*-
"""
Halo Phase 4: HALO Sigma Model Learning

Learns operator-level hardware sensitivity vectors:
  sigma = [sigma_storage, sigma_compute]

Core idea: Match the same operator instance across different environments,
compute Delta_time = log(self_time_env2 / self_time_env1),
and train a regression model to predict Delta_time from operator features
and environment-pair indicators.

Model: XGBoost regressor (CPU only, ~minutes to train)
"""

import pandas as pd
import numpy as np
import json
import os
import math
import logging
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Environment definitions
ENV_MAP = {
    'A_NVMe': {'server': 'A', 'storage': 'NVMe', 'cpu': 'i9'},
    'A_SATA': {'server': 'A', 'storage': 'SATA', 'cpu': 'i9'},
    'B_NVMe': {'server': 'B', 'storage': 'NVMe', 'cpu': 'EPYC'},
    'B_SATA': {'server': 'B', 'storage': 'SATA', 'cpu': 'EPYC'},
}

# Environment pairs and their factor changes
ENV_PAIRS = []
for e1 in sorted(ENV_MAP.keys()):
    for e2 in sorted(ENV_MAP.keys()):
        if e1 < e2:
            storage_changed = ENV_MAP[e1]['storage'] != ENV_MAP[e2]['storage']
            compute_changed = ENV_MAP[e1]['cpu'] != ENV_MAP[e2]['cpu']
            ENV_PAIRS.append({
                'env1': e1, 'env2': e2,
                'storage_changed': storage_changed,
                'compute_changed': compute_changed,
                'pair_label': f"{e1}_vs_{e2}"
            })


def build_env_pairs_dataset(df_ops):
    """
    Match same (query_id, hint_variant, hint_set, tree_position) across environments.
    Compute Delta_time = log(self_time_env2 / self_time_env1) for each pair.

    Returns DataFrame with ~30K+ training samples.
    """
    logger.info("Building environment-pair dataset...")

    # Create matching key
    df_ops = df_ops.copy()
    df_ops['match_key'] = (
        df_ops['query_id'] + '|' +
        df_ops['hint_variant'] + '|' +
        df_ops['hint_set'] + '|' +
        df_ops['tree_position'].astype(str) + '|' +
        df_ops['benchmark']
    )

    samples = []
    for pair in ENV_PAIRS:
        e1, e2 = pair['env1'], pair['env2']

        ops1 = df_ops[df_ops['env_id'] == e1].set_index('match_key')
        ops2 = df_ops[df_ops['env_id'] == e2].set_index('match_key')

        # Find matching operators
        common_keys = ops1.index.intersection(ops2.index)
        if len(common_keys) == 0:
            logger.info(f"  {e1} vs {e2}: 0 matches (no overlap)")
            continue

        for key in common_keys:
            try:
                o1 = ops1.loc[key]
                o2 = ops2.loc[key]

                # Handle duplicates (take first if multiple)
                if isinstance(o1, pd.DataFrame):
                    o1 = o1.iloc[0]
                if isinstance(o2, pd.DataFrame):
                    o2 = o2.iloc[0]

                # Must be same operator type for meaningful comparison
                if o1['op_type'] != o2['op_type']:
                    continue

                st1 = max(o1['self_time'], 0.001)
                st2 = max(o2['self_time'], 0.001)

                delta_time = math.log(st2 / st1)

                samples.append({
                    # Identity
                    'match_key': key,
                    'query_id': o1['query_id'],
                    'benchmark': o1['benchmark'],
                    'hint_set': o1['hint_set'],
                    'env1': e1,
                    'env2': e2,
                    'pair_label': pair['pair_label'],
                    # Environment factors (2D sigma)
                    'storage_changed': int(pair['storage_changed']),
                    'compute_changed': int(pair['compute_changed']),
                    # Operator features (from env1 as reference)
                    'op_type': o1['op_type'],
                    'log_actual_rows': math.log(max(o1['actual_rows'], 1)),
                    'log_loops': math.log(max(o1['loops'], 1)),
                    'row_est_error': o1['row_est_error'],
                    'depth': o1['depth'],
                    'cost_per_row': o1['cost_per_row'],
                    'parent_op_type': o1['parent_op_type'],
                    'self_time_env1': st1,
                    'self_time_env2': st2,
                    # Target
                    'delta_time': delta_time,
                })

            except Exception:
                continue

        logger.info(f"  {e1} vs {e2}: {sum(1 for s in samples if s['pair_label'] == pair['pair_label']):,} matches")

    df_samples = pd.DataFrame(samples)
    logger.info(f"Total training samples: {len(df_samples):,}")
    return df_samples


def prepare_features(df_samples):
    """
    Prepare feature matrix X and target y for sigma model training.
    """
    # Encode categorical features
    le_op = LabelEncoder()
    le_parent = LabelEncoder()

    df = df_samples.copy()
    df['op_type_enc'] = le_op.fit_transform(df['op_type'].fillna('UNKNOWN'))
    df['parent_op_type_enc'] = le_parent.fit_transform(df['parent_op_type'].fillna('NONE'))

    feature_cols = [
        'op_type_enc',
        'log_actual_rows',
        'log_loops',
        'row_est_error',
        'depth',
        'cost_per_row',
        'parent_op_type_enc',
        'storage_changed',
        'compute_changed',
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df['delta_time'].values.astype(np.float32)

    # Clean NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    df_clean = df[mask].reset_index(drop=True)

    return X, y, df_clean, feature_cols, le_op, le_parent


def train_sigma_model(X, y, df_clean, feature_cols):
    """
    Train XGBoost regressor for sigma prediction.
    Evaluation: Leave-1-env-pair-out cross-validation.
    """
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False

    from sklearn.ensemble import GradientBoostingRegressor

    logger.info(f"\nTraining sigma model...")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Target (delta_time) stats: mean={y.mean():.3f}, std={y.std():.3f}")

    # ── Model ──
    if has_xgb:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            tree_method='hist',
            random_state=42,
        )
        model_name = 'XGBoost'
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model_name = 'GBT (sklearn)'

    logger.info(f"  Model: {model_name}")

    # ── Leave-1-env-pair-out CV ──
    groups = df_clean['pair_label'].values
    logo = LeaveOneGroupOut()

    y_pred_cv = np.zeros_like(y)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_pair = groups[test_idx[0]]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_cv[test_idx] = y_pred

        mape = np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + 1e-6)) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_results.append({
            'pair': test_pair,
            'n_test': len(test_idx),
            'mape': mape,
            'rmse': rmse,
        })
        logger.info(f"  Fold {fold_idx+1} ({test_pair}): MAPE={mape:.1f}%, RMSE={rmse:.3f}, n={len(test_idx)}")

    # ── Overall metrics ──
    overall_mape = np.mean(np.abs(y - y_pred_cv) / (np.abs(y) + 1e-6)) * 100
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_cv))

    # Ranking accuracy: for each operator, is the direction (faster/slower) correct?
    direction_correct = ((y > 0) == (y_pred_cv > 0)) | (np.abs(y) < 0.1)
    ranking_acc = direction_correct.mean() * 100

    # Correlation
    corr = np.corrcoef(y, y_pred_cv)[0, 1]

    logger.info(f"\n{'='*60}")
    logger.info(f"Overall Leave-1-env-pair-out Results:")
    logger.info(f"  MAPE: {overall_mape:.1f}%")
    logger.info(f"  RMSE: {overall_rmse:.3f}")
    logger.info(f"  Ranking accuracy: {ranking_acc:.1f}%")
    logger.info(f"  Pearson r: {corr:.3f}")

    # ── Train final model on all data ──
    model.fit(X, y)

    # ── Feature importance ──
    if has_xgb:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    feat_imp = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1], reverse=True
    )
    logger.info(f"\nFeature Importance:")
    for fname, imp in feat_imp:
        logger.info(f"  {fname}: {imp:.3f}")

    results = {
        'model_name': model_name,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'overall_mape': float(overall_mape),
        'overall_rmse': float(overall_rmse),
        'ranking_accuracy': float(ranking_acc),
        'pearson_r': float(corr),
        'fold_results': fold_results,
        'feature_importance': {k: float(v) for k, v in feat_imp},
        'feature_cols': feature_cols,
    }

    return model, results, y_pred_cv


def analyze_sigma_by_operator(df_clean, y_pred_cv):
    """
    Analyze learned sigma values per operator type.
    Shows which operators are most hardware-sensitive.
    """
    df = df_clean.copy()
    df['predicted_delta'] = y_pred_cv
    df['abs_delta'] = np.abs(df['delta_time'])
    df['abs_predicted'] = np.abs(y_pred_cv)

    logger.info(f"\nSigma Analysis by Operator Type:")
    logger.info(f"{'Op Type':<25} {'Count':>8} {'Mean |delta|':>12} {'Pred |delta|':>12} {'Storage Sens':>12} {'Compute Sens':>12}")
    logger.info("-" * 85)

    op_stats = []
    for op_type, group in df.groupby('op_type'):
        storage_group = group[group['storage_changed'] == 1]
        compute_group = group[group['compute_changed'] == 1]

        storage_sens = storage_group['abs_delta'].mean() if len(storage_group) > 0 else 0
        compute_sens = compute_group['abs_delta'].mean() if len(compute_group) > 0 else 0

        logger.info(f"{op_type:<25} {len(group):>8,} {group['abs_delta'].mean():>12.3f} "
                    f"{group['abs_predicted'].mean():>12.3f} {storage_sens:>12.3f} {compute_sens:>12.3f}")

        op_stats.append({
            'op_type': op_type,
            'count': len(group),
            'mean_abs_delta': float(group['abs_delta'].mean()),
            'sigma_storage': float(storage_sens),
            'sigma_compute': float(compute_sens),
        })

    return op_stats


def run_sigma_learning(data_dir='/root/halo/data', output_dir='/root/halo/results'):
    """Full sigma learning pipeline."""
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    # Load
    df_ops = pd.read_parquet(os.path.join(data_dir, 'unified_operators.parquet'))
    logger.info(f"Loaded {len(df_ops):,} operator instances from {df_ops['env_id'].nunique()} envs")

    # Build env-pair dataset
    df_samples = build_env_pairs_dataset(df_ops)

    if len(df_samples) == 0:
        logger.error("No matching samples found across environments!")
        return

    # Save samples
    df_samples.to_parquet(os.path.join(output_dir, 'sigma_training_samples.parquet'), index=False)

    # Prepare features
    X, y, df_clean, feature_cols, le_op, le_parent = prepare_features(df_samples)

    # Train
    model, results, y_pred_cv = train_sigma_model(X, y, df_clean, feature_cols)

    # Analyze
    op_stats = analyze_sigma_by_operator(df_clean, y_pred_cv)
    results['operator_sigma'] = op_stats

    # Save model
    with open(os.path.join(output_dir, 'sigma_model.pkl'), 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'le_op': le_op,
            'le_parent': le_parent,
        }, f)

    # Save results
    with open(os.path.join(output_dir, 'sigma_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save predictions for analysis
    df_clean['predicted_delta'] = y_pred_cv
    df_clean.to_parquet(os.path.join(output_dir, 'sigma_predictions.parquet'), index=False)

    logger.info(f"\nAll results saved to {output_dir}/")
    return model, results


if __name__ == '__main__':
    model, results = run_sigma_learning()
    print("\n" + json.dumps({
        'mape': results['overall_mape'],
        'rmse': results['overall_rmse'],
        'ranking_accuracy': results['ranking_accuracy'],
        'pearson_r': results['pearson_r'],
        'n_samples': results['n_samples'],
    }, indent=2))
