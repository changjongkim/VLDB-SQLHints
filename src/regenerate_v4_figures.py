#!/usr/bin/env python3
"""
HALO v4 — Full Model Explainability Figure Generator
=====================================================
Re-runs v4 LOGO-CV from scratch using unified_operators.parquet,
saves v4 predictions, and generates all §6.5 + §10.3 figures.

Figures generated:
  1. fig_sigma_evaluation.png       — 4-panel σ model evaluation
  2. fig_sigma_feature_importance.png — Feature importance bar chart
  3. fig_sigma_density.png          — σ-accuracy density hexbin
  4. fig_explain_accuracy_operator.png — Precision/Recall/F1 by operator
  5. fig_explain_accuracy_env.png   — Classification accuracy by env pair
  6. fig_explain_decision_logic.png — Decision logic scatter
  7. fig_explain_risky_operators.png — Safety fallback triggers
  8. fig_explain_confusion_matrix.png — Overall confusion matrix
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

# Add src to path
sys.path.insert(0, '/root/halo/src')
from sigma_model_v4 import (
    HALOProbabilisticMLP, GaussianNLLLoss, mc_dropout_predict,
    ConformalCalibrator, build_env_pairs_v4
)
from risk_pattern_dictionary import RiskPatternDictionary

OUT_DIR = '/root/halo/assets'
RESULTS_DIR = '/root/halo/results/v4'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Plot style ──
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'DejaVu Sans',
})
COLORS = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'success': '#059669',
    'danger': '#DC2626',
    'warning': '#D97706',
    'info': '#0891B2',
    'gray': '#6B7280',
    'dark': '#1F2937',
}

# ===================================================================
#  STEP 1: Rebuild v4 dataset & run LOGO-CV
# ===================================================================
def run_v4_cv():
    """Run the full v4 Leave-One-Group-Out CV and save predictions."""
    print("=" * 60)
    print("STEP 1: Building v4 dataset from unified_operators...")
    print("=" * 60)

    df_ops = pd.read_parquet('/root/halo/data/unified_operators.parquet')
    print(f"  Loaded {len(df_ops):,} operators")

    risk_dict = RiskPatternDictionary()
    df_samples, risk_dict = build_env_pairs_v4(df_ops, risk_dict)

    # Save v4 training data
    df_samples.to_parquet(os.path.join(RESULTS_DIR, 'sigma_training_v4.parquet'), index=False)
    print(f"  Training samples: {len(df_samples):,}")

    # Identify columns
    meta_cols = ['delta_time', 'self_time_env1', 'self_time_env2',
                 'query_id', 'benchmark', 'hint_set', 'op_type',
                 'env1', 'env2', 'pair_label']
    feature_cols = [c for c in df_samples.columns if c not in meta_cols]

    X = df_samples[feature_cols].values.astype(np.float32)
    y = df_samples['delta_time'].values.astype(np.float32)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    df_clean = df_samples[mask].reset_index(drop=True)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Clean samples: {len(X):,}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sample weights
    pair_counts = df_clean['pair_label'].value_counts()
    max_count = pair_counts.max()
    env_weights = df_clean['pair_label'].map(lambda p: max_count / pair_counts.get(p, 1)).values
    risk_weights = np.ones_like(env_weights)
    risk_weights[y > 0.8] *= 2.0
    high_risk_mask = np.abs(y) > 0.5
    risk_weights[high_risk_mask] *= 3.0
    final_weights = env_weights * risk_weights
    final_weights = final_weights / final_weights.mean()

    # LOGO-CV
    groups = df_clean['pair_label'].values
    logo = LeaveOneGroupOut()

    y_mu_cv = np.zeros_like(y)
    y_sigma_cv = np.zeros_like(y)
    fold_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"\n  Running Leave-One-Group-Out CV...")

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
        X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1).to(device)
        w_train = torch.FloatTensor(final_weights[train_idx]).to(device)
        X_test = torch.FloatTensor(X_scaled[test_idx]).to(device)

        model = HALOProbabilisticMLP(
            input_dim=X_scaled.shape[1],
            hidden_dims=[256, 128, 64],
            dropout_rate=0.1,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = GaussianNLLLoss()

        dataset = TensorDataset(X_train, y_train, w_train)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)

        model.train()
        for epoch in range(100):
            for X_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                mu, sigma = model(X_batch)
                loss = criterion(mu, sigma, y_batch, w_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # MC-Dropout prediction
        mu_pred, sigma_pred = mc_dropout_predict(model, X_test, n_samples=30)
        y_mu_cv[test_idx] = mu_pred
        y_sigma_cv[test_idx] = sigma_pred

        pair_name = groups[test_idx[0]]
        rmse = np.sqrt(mean_squared_error(y[test_idx], mu_pred))
        r2 = r2_score(y[test_idx], mu_pred) if len(test_idx) > 1 else 0
        nonzero = np.abs(y[test_idx]) > 0.1
        dir_acc = ((y[test_idx][nonzero] > 0) == (mu_pred[nonzero] > 0)).mean() * 100 \
                  if nonzero.sum() > 0 else 100.0

        fold_results.append({
            'pair': pair_name, 'n': len(test_idx),
            'rmse': float(rmse), 'r2': float(r2), 'dir_acc': float(dir_acc)
        })
        print(f"    Fold {fold_idx+1} ({pair_name}): RMSE={rmse:.3f}, R²={r2:.3f}, DirAcc={dir_acc:.1f}%")

    # Overall
    overall_rmse = np.sqrt(mean_squared_error(y, y_mu_cv))
    overall_r2 = r2_score(y, y_mu_cv)
    overall_corr = np.corrcoef(y, y_mu_cv)[0, 1]
    overall_dir_acc = 0.0
    nonzero_mask = np.abs(y) > 0.1
    if nonzero_mask.sum() > 0:
        overall_dir_acc = ((y[nonzero_mask] > 0) == (y_mu_cv[nonzero_mask] > 0)).mean() * 100

    # Conformal calibration
    calibrator = ConformalCalibrator(alpha=0.10)
    lambda_cal = calibrator.calibrate(y, y_mu_cv, y_sigma_cv)

    print(f"\n  Overall: RMSE={overall_rmse:.3f}, R²={overall_r2:.3f}, "
          f"r={overall_corr:.3f}, DirAcc={overall_dir_acc:.1f}%")
    print(f"  Conformal λ={lambda_cal:.3f}")

    # Save predictions
    df_pred = df_clean.copy()
    df_pred['predicted_mu'] = y_mu_cv
    df_pred['predicted_sigma'] = y_sigma_cv
    df_pred['upper_bound'] = y_mu_cv + lambda_cal * y_sigma_cv
    df_pred.to_parquet(os.path.join(RESULTS_DIR, 'sigma_predictions_v4.parquet'), index=False)
    print(f"  Predictions saved: {RESULTS_DIR}/sigma_predictions_v4.parquet")

    # Save results JSON
    results = {
        'model_version': 'v4_conformal',
        'n_samples': int(len(X)),
        'n_features': len(feature_cols),
        'feature_list': feature_cols,
        'overall_rmse': float(overall_rmse),
        'overall_r2': float(overall_r2),
        'pearson_r': float(overall_corr),
        'direction_accuracy': float(overall_dir_acc),
        'conformal_lambda': float(lambda_cal),
        'mean_sigma': float(y_sigma_cv.mean()),
        'fold_results': fold_results,
    }
    with open(os.path.join(RESULTS_DIR, 'sigma_results_v4_cv.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return df_pred, feature_cols, results, scaler


# ===================================================================
#  FIGURE 1: σ-Model 4-Panel Evaluation
# ===================================================================
def fig_sigma_evaluation(df):
    """4-panel σ model evaluation."""
    print("\n  Generating: fig_sigma_evaluation.png")
    y = df['delta_time'].values
    mu = df['predicted_mu'].values
    sigma = df['predicted_sigma'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('HALO v4 σ-Model Evaluation (LOGO-CV)', fontsize=16, fontweight='bold', y=0.98)

    # (a) Predicted vs Actual
    ax = axes[0, 0]
    hb = ax.hexbin(y, mu, gridsize=50, cmap='Blues', mincnt=1, linewidths=0.2)
    lims = [min(y.min(), mu.min()), max(y.max(), mu.max())]
    ax.plot(lims, lims, 'r--', lw=2, alpha=0.8, label='Perfect')
    r2 = r2_score(y, mu)
    corr = np.corrcoef(y, mu)[0, 1]
    ax.set_xlabel('Actual δ(time)')
    ax.set_ylabel('Predicted μ')
    ax.set_title(f'(a) Predicted vs Actual\nR²={r2:.3f}, Pearson r={corr:.3f}')
    ax.legend()
    plt.colorbar(hb, ax=ax, label='Count')

    # (b) Residual distribution
    ax = axes[0, 1]
    residuals = y - mu
    ax.hist(residuals, bins=80, color=COLORS['primary'], alpha=0.7, edgecolor='white', density=True)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    rmse = np.sqrt(mean_squared_error(y, mu))
    ax.set_xlabel('Residual (Actual − Predicted)')
    ax.set_ylabel('Density')
    ax.set_title(f'(b) Residual Distribution\nRMSE={rmse:.3f}, Mean={residuals.mean():.3f}')

    # (c) σ vs |Residual| (uncertainty calibration)
    ax = axes[1, 0]
    abs_res = np.abs(residuals)
    hb2 = ax.hexbin(sigma, abs_res, gridsize=40, cmap='Oranges', mincnt=1, linewidths=0.2)
    # bin σ and plot mean |residual|
    n_bins = 20
    sigma_bins = np.linspace(sigma.min(), np.percentile(sigma, 95), n_bins + 1)
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        mask = (sigma >= sigma_bins[i]) & (sigma < sigma_bins[i+1])
        if mask.sum() > 10:
            bin_centers.append((sigma_bins[i] + sigma_bins[i+1]) / 2)
            bin_means.append(abs_res[mask].mean())
    ax.plot(bin_centers, bin_means, 'r-o', lw=2, ms=5, label='Binned Mean |Res|')
    ax.plot([0, max(bin_centers)], [0, max(bin_centers)], 'k--', alpha=0.5, label='Perfect Cal.')
    ax.set_xlabel('Predicted σ (Uncertainty)')
    ax.set_ylabel('|Residual|')
    ax.set_title('(c) Uncertainty Calibration\nHigher σ → Larger Error (good!)')
    ax.legend(fontsize=9)
    plt.colorbar(hb2, ax=ax, label='Count')

    # (d) Direction accuracy by env pair
    ax = axes[1, 1]
    pairs = df['pair_label'].unique()
    pair_stats = []
    for p in sorted(pairs):
        m = df['pair_label'] == p
        yt = y[m]
        ym = mu[m]
        nz = np.abs(yt) > 0.1
        if nz.sum() > 0:
            da = ((yt[nz] > 0) == (ym[nz] > 0)).mean() * 100
        else:
            da = 100.0
        pair_stats.append({'pair': p.replace('_vs_', '\nvs\n'), 'dir_acc': da, 'n': int(m.sum())})

    pdf = pd.DataFrame(pair_stats)
    bars = ax.barh(pdf['pair'], pdf['dir_acc'], color=COLORS['secondary'], alpha=0.85, edgecolor='white')
    for bar, row in zip(bars, pdf.itertuples()):
        ax.text(bar.get_width() - 3, bar.get_y() + bar.get_height()/2,
                f'{row.dir_acc:.1f}%', va='center', ha='right', fontsize=9, fontweight='bold', color='white')
    ax.set_xlabel('Direction Accuracy (%)')
    ax.set_title(f'(d) Direction Accuracy by HW Pair\nOverall: {((y[np.abs(y)>0.1]>0)==(mu[np.abs(y)>0.1]>0)).mean()*100:.1f}%')
    ax.set_xlim(50, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, 'fig_sigma_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 2: Feature Importance
# ===================================================================
def fig_feature_importance(df, feature_cols, scaler):
    """Feature importance via permutation-like correlation approach."""
    print("\n  Generating: fig_sigma_feature_importance.png")
    y = df['delta_time'].values
    mu = df['predicted_mu'].values

    # Use correlation between each feature and |residual| as importance proxy
    residuals = np.abs(y - mu)
    X = df[feature_cols].values
    importances = []
    for i, col in enumerate(feature_cols):
        x_col = X[:, i]
        finite_mask = np.isfinite(x_col)
        if finite_mask.sum() > 100:
            corr = np.abs(np.corrcoef(x_col[finite_mask], residuals[finite_mask])[0, 1])
        else:
            corr = 0.0
        importances.append({'feature': col, 'importance': corr if np.isfinite(corr) else 0.0})

    # Also add: correlation with sigma (uncertainty)
    sigma = df['predicted_sigma'].values
    sigma_importances = []
    for i, col in enumerate(feature_cols):
        x_col = X[:, i]
        finite_mask = np.isfinite(x_col)
        if finite_mask.sum() > 100:
            corr = np.abs(np.corrcoef(x_col[finite_mask], sigma[finite_mask])[0, 1])
        else:
            corr = 0.0
        sigma_importances.append({'feature': col, 'sigma_corr': corr if np.isfinite(corr) else 0.0})

    imp_df = pd.DataFrame(importances).merge(pd.DataFrame(sigma_importances), on='feature')
    imp_df['combined'] = imp_df['importance'] * 0.5 + imp_df['sigma_corr'] * 0.5
    imp_df = imp_df.sort_values('combined', ascending=True).tail(25)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('HALO v4 Feature Importance Analysis', fontsize=15, fontweight='bold')

    # Left: correlation with |residual|
    ax = axes[0]
    colors_left = [COLORS['danger'] if 'is_' in f and 'JOIN' in f else
                   COLORS['primary'] if 'ratio' in f or 'changed' in f else
                   COLORS['success'] if 'alignment' in f or 'pattern' in f else
                   COLORS['warning']
                   for f in imp_df['feature']]
    ax.barh(imp_df['feature'], imp_df['importance'], color=colors_left, alpha=0.85, edgecolor='white')
    ax.set_xlabel('|Correlation| with |Residual|')
    ax.set_title('(a) Prediction Error Drivers')

    # Right: correlation with σ (what drives uncertainty)
    ax = axes[1]
    imp_sigma = imp_df.sort_values('sigma_corr', ascending=True)
    colors_right = [COLORS['secondary'] if 'sigma' in f or 'alignment' in f or 'pattern' in f else
                    COLORS['info'] for f in imp_sigma['feature']]
    ax.barh(imp_sigma['feature'], imp_sigma['sigma_corr'], color=colors_right, alpha=0.85, edgecolor='white')
    ax.set_xlabel('|Correlation| with σ (Uncertainty)')
    ax.set_title('(b) Uncertainty Drivers')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_sigma_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 3: σ Density vs Accuracy
# ===================================================================
def fig_sigma_density(df):
    """σ-accuracy density visualization."""
    print("\n  Generating: fig_sigma_density.png")
    y = df['delta_time'].values
    mu = df['predicted_mu'].values
    sigma = df['predicted_sigma'].values
    abs_error = np.abs(y - mu)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('HALO v4: Uncertainty–Error Relationship', fontsize=15, fontweight='bold')

    # (a) Hexbin: σ vs absolute error
    ax = axes[0]
    hb = ax.hexbin(sigma, abs_error, gridsize=40, cmap='YlOrRd', mincnt=1, linewidths=0.1)
    ax.set_xlabel('Predicted σ')
    ax.set_ylabel('|Actual − Predicted|')
    ax.set_title('(a) σ vs Prediction Error')
    plt.colorbar(hb, ax=ax, label='Density')

    # Add diagonal reference
    max_s = np.percentile(sigma, 95)
    ax.plot([0, max_s], [0, max_s], 'k--', alpha=0.5, lw=1.5)
    ax.set_xlim(0, max_s)
    ax.set_ylim(0, np.percentile(abs_error, 97))

    # (b) σ distribution colored by correctness
    ax = axes[1]
    correct_mask = np.abs(y) > 0.1
    correct = ((y[correct_mask] > 0) == (mu[correct_mask] > 0))
    sigma_correct = sigma[correct_mask][correct]
    sigma_wrong = sigma[correct_mask][~correct]

    ax.hist(sigma_correct, bins=50, alpha=0.6, color=COLORS['success'], label=f'Correct ({len(sigma_correct):,})',
            density=True, edgecolor='white')
    ax.hist(sigma_wrong, bins=50, alpha=0.6, color=COLORS['danger'], label=f'Wrong ({len(sigma_wrong):,})',
            density=True, edgecolor='white')
    ax.set_xlabel('Predicted σ')
    ax.set_ylabel('Density')
    ax.set_title('(b) σ Distribution: Correct vs Wrong')
    ax.legend()
    # Add median lines
    ax.axvline(np.median(sigma_correct), color=COLORS['success'], linestyle='--', lw=2, alpha=0.8)
    ax.axvline(np.median(sigma_wrong), color=COLORS['danger'], linestyle='--', lw=2, alpha=0.8)

    # (c) Binned accuracy by σ quantile
    ax = axes[2]
    n_bins = 10
    sigma_quantiles = np.quantile(sigma[correct_mask], np.linspace(0, 1, n_bins + 1))
    bin_accs = []
    bin_labels = []
    for i in range(n_bins):
        m = (sigma[correct_mask] >= sigma_quantiles[i]) & (sigma[correct_mask] < sigma_quantiles[i+1])
        if m.sum() > 0:
            acc = correct[m].mean() * 100
            bin_accs.append(acc)
            bin_labels.append(f'Q{i+1}\n({sigma_quantiles[i]:.2f}-{sigma_quantiles[i+1]:.2f})')
    colors = [COLORS['success'] if a > 80 else COLORS['warning'] if a > 65 else COLORS['danger'] for a in bin_accs]
    bars = ax.bar(bin_labels, bin_accs, color=colors, alpha=0.85, edgecolor='white')
    for bar, acc in zip(bars, bin_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('σ Quantile (Low → High Uncertainty)')
    ax.set_ylabel('Direction Accuracy (%)')
    ax.set_title('(c) Lower σ → Higher Accuracy')
    ax.set_ylim(50, 100)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_sigma_density.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 4: High-Risk Detection by Operator Type
# ===================================================================
def fig_accuracy_by_operator(df, results):
    """Precision / Recall / F1 for high-risk detection by operator type."""
    print("\n  Generating: fig_explain_accuracy_operator.png")
    y = df['delta_time'].values
    ub = df['upper_bound'].values
    op_types = df['op_type'].values

    threshold = 0.5
    y_high = np.abs(y) > threshold
    y_pred_high = ub > threshold

    unique_ops = sorted(df['op_type'].unique())
    op_metrics = []

    for op in unique_ops:
        m = op_types == op
        if y_high[m].sum() < 5:
            continue
        prec = precision_score(y_high[m], y_pred_high[m], zero_division=0)
        rec = recall_score(y_high[m], y_pred_high[m], zero_division=0)
        f1 = f1_score(y_high[m], y_pred_high[m], zero_division=0)
        n_total = m.sum()
        n_high = y_high[m].sum()
        op_metrics.append({
            'op_type': op, 'precision': prec, 'recall': rec, 'f1': f1,
            'n_total': n_total, 'n_high': n_high, 'risk_rate': n_high / n_total
        })

    op_df = pd.DataFrame(op_metrics).sort_values('f1', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'HALO v4: High-Risk Detection by Operator Type (δ > {threshold})',
                 fontsize=15, fontweight='bold')

    # Left: P/R/F1
    ax = axes[0]
    x = np.arange(len(op_df))
    w = 0.25
    ax.barh(x - w, op_df['precision'], w, label='Precision', color=COLORS['primary'], alpha=0.85)
    ax.barh(x, op_df['recall'], w, label='Recall', color=COLORS['danger'], alpha=0.85)
    ax.barh(x + w, op_df['f1'], w, label='F1', color=COLORS['success'], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(op_df['op_type'], fontsize=9)
    ax.set_xlabel('Score')
    ax.set_title('(a) Precision / Recall / F1 per Operator')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.05)

    # Right: Risk rate + sample count
    ax = axes[1]
    bars = ax.barh(op_df['op_type'], op_df['risk_rate'] * 100,
                   color=[COLORS['danger'] if r > 0.3 else COLORS['warning'] if r > 0.15 else COLORS['success']
                          for r in op_df['risk_rate']], alpha=0.85, edgecolor='white')
    for bar, row in zip(bars, op_df.itertuples()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'n={row.n_total:,} ({row.n_high} risky)', va='center', fontsize=8)
    ax.set_xlabel('Risk Rate (%)')
    ax.set_title('(b) Operator Risk Rate (% of samples with |δ|>0.5)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_explain_accuracy_operator.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 5: Classification Accuracy by Env Pair
# ===================================================================
def fig_accuracy_by_env(df, results):
    """Classification metrics grouped by hardware environment pair."""
    print("\n  Generating: fig_explain_accuracy_env.png")
    y = df['delta_time'].values
    mu = df['predicted_mu'].values
    sigma = df['predicted_sigma'].values
    ub = df['upper_bound'].values

    pairs = sorted(df['pair_label'].unique())
    pair_metrics = []

    for p in pairs:
        m = df['pair_label'] == p
        yt, ym, ys = y[m], mu[m], sigma[m]

        rmse = np.sqrt(mean_squared_error(yt, ym))
        r2 = r2_score(yt, ym) if len(yt) > 1 else 0

        nz = np.abs(yt) > 0.1
        dir_acc = ((yt[nz] > 0) == (ym[nz] > 0)).mean() * 100 if nz.sum() > 0 else 100

        # High risk detection
        y_high = np.abs(yt) > 0.5
        y_pred_high = ub[m] > 0.5
        prec = precision_score(y_high, y_pred_high, zero_division=0)
        rec = recall_score(y_high, y_pred_high, zero_division=0)
        f1 = f1_score(y_high, y_pred_high, zero_division=0)

        pair_metrics.append({
            'pair': p, 'n': int(m.sum()),
            'rmse': rmse, 'r2': r2, 'dir_acc': dir_acc,
            'mean_sigma': float(ys.mean()),
            'precision': prec, 'recall': rec, 'f1': f1
        })

    pdf = pd.DataFrame(pair_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HALO v4: Performance by Hardware Environment Pair', fontsize=15, fontweight='bold')

    # (a) Direction accuracy
    ax = axes[0, 0]
    bars = ax.bar(pdf['pair'], pdf['dir_acc'], color=COLORS['primary'], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, pdf['dir_acc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Direction Accuracy (%)')
    ax.set_title('(a) Direction Accuracy')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(50, 100)

    # (b) RMSE
    ax = axes[0, 1]
    bars = ax.bar(pdf['pair'], pdf['rmse'], color=COLORS['warning'], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, pdf['rmse']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('RMSE')
    ax.set_title('(b) Prediction Error (RMSE)')
    ax.tick_params(axis='x', rotation=45)

    # (c) F1 for high-risk detection
    ax = axes[1, 0]
    x = np.arange(len(pdf))
    w = 0.25
    ax.bar(x - w, pdf['precision'], w, label='Precision', color=COLORS['primary'], alpha=0.85)
    ax.bar(x, pdf['recall'], w, label='Recall', color=COLORS['danger'], alpha=0.85)
    ax.bar(x + w, pdf['f1'], w, label='F1', color=COLORS['success'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(pdf['pair'], fontsize=8, rotation=45)
    ax.set_ylabel('Score')
    ax.set_title('(c) High-Risk Detection (|δ|>0.5)')
    ax.legend(fontsize=8)

    # (d) Mean σ
    ax = axes[1, 1]
    bars = ax.bar(pdf['pair'], pdf['mean_sigma'], color=COLORS['secondary'], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, pdf['mean_sigma']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean σ')
    ax.set_title('(d) Average Uncertainty by Pair')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_explain_accuracy_env.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 6: Decision Logic Visualization
# ===================================================================
def fig_decision_logic(df, results):
    """HALO-R decision logic: σ threshold scatter."""
    print("\n  Generating: fig_explain_decision_logic.png")
    y = df['delta_time'].values
    mu = df['predicted_mu'].values
    sigma = df['predicted_sigma'].values
    ub = df['upper_bound'].values
    lambda_cal = results.get('conformal_lambda', 2.146)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('HALO v4: Decision Logic — When to RECOMMEND vs NATIVE',
                 fontsize=15, fontweight='bold')

    # (a) Upper bound vs actual delta
    ax = axes[0]
    # Color by decision
    thresh = 0.1  # recommend threshold
    recommend = ub <= thresh
    native = ub > thresh

    # Among recommended, separate safe (actual <= 0.5) vs regressed
    rec_safe = recommend & (y < 0.5)
    rec_regress = recommend & (y >= 0.5)
    nat_correct = native & (y >= 0.0)
    nat_missed = native & (y < 0.0)

    ax.scatter(mu[rec_safe], sigma[rec_safe], s=3, alpha=0.2, c=COLORS['success'], label=f'RECOMMEND safe ({rec_safe.sum():,})')
    ax.scatter(mu[rec_regress], sigma[rec_regress], s=8, alpha=0.5, c=COLORS['danger'], label=f'RECOMMEND regress ({rec_regress.sum():,})', marker='x')
    ax.scatter(mu[nat_correct], sigma[nat_correct], s=3, alpha=0.15, c=COLORS['gray'], label=f'NATIVE correct ({nat_correct.sum():,})')
    ax.scatter(mu[nat_missed], sigma[nat_missed], s=5, alpha=0.3, c=COLORS['warning'], label=f'NATIVE missed ({nat_missed.sum():,})')

    ax.set_xlabel('Predicted μ (expected delta)')
    ax.set_ylabel('Predicted σ (uncertainty)')
    ax.set_title(f'(a) Decision Space (λ={lambda_cal:.3f}, threshold={thresh})')
    ax.legend(fontsize=8, loc='upper right', markerscale=3)

    # (b) Safety metrics by σ quantile
    ax = axes[1]
    n_bins = 8
    sigma_q = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    bin_safety = []
    bin_labels = []
    for i in range(n_bins):
        m = (sigma >= sigma_q[i]) & (sigma < sigma_q[i+1])
        if m.sum() > 0:
            # Safety = fraction where recommended hints don't cause regression
            rec_m = recommend & m
            if rec_m.sum() > 0:
                safety_rate = (y[rec_m] < 0.5).mean() * 100
            else:
                safety_rate = 100.0
            bin_safety.append(safety_rate)
            bin_labels.append(f'σ∈[{sigma_q[i]:.2f},{sigma_q[i+1]:.2f})')

    colors = [COLORS['success'] if s > 90 else COLORS['warning'] if s > 75 else COLORS['danger'] for s in bin_safety]
    bars = ax.bar(bin_labels, bin_safety, color=colors, alpha=0.85, edgecolor='white')
    for bar, s in zip(bars, bin_safety):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{s:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('σ Range')
    ax.set_ylabel('Safety Rate (% recommendations without regression)')
    ax.set_title('(b) Safety Rate by Uncertainty Level')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(50, 105)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% target')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_explain_decision_logic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 7: Safety Fallback Trigger (Risky Operators)
# ===================================================================
def fig_risky_operators(df, results):
    """Which operators trigger safety fallbacks most often?"""
    print("\n  Generating: fig_explain_risky_operators.png")
    y = df['delta_time'].values
    ub = df['upper_bound'].values
    op_types = df['op_type'].values

    # Fallback = upper_bound > threshold (NATIVE decision)
    threshold = 0.1
    fallback = ub > threshold
    actual_regress = y > 0.5

    op_fallback = []
    for op in sorted(df['op_type'].unique()):
        m = op_types == op
        n_total = m.sum()
        n_fallback = fallback[m].sum()
        n_actual_regress = actual_regress[m].sum()
        # True positive fallback: operator flagged AND actually dangerous
        tp_fallback = (fallback[m] & actual_regress[m]).sum()

        op_fallback.append({
            'op_type': op,
            'n_total': n_total,
            'n_fallback': n_fallback,
            'fallback_rate': n_fallback / max(n_total, 1) * 100,
            'actual_risk_rate': n_actual_regress / max(n_total, 1) * 100,
            'tp_rate': tp_fallback / max(n_fallback, 1) * 100 if n_fallback > 0 else 0,
        })

    opdf = pd.DataFrame(op_fallback).sort_values('fallback_rate', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('HALO v4: Safety Fallback Triggers by Operator Type',
                 fontsize=15, fontweight='bold')

    x = np.arange(len(opdf))
    w = 0.35
    bars1 = ax.barh(x - w/2, opdf['fallback_rate'], w, label='Fallback Rate (NATIVE decision)',
                    color=COLORS['warning'], alpha=0.85, edgecolor='white')
    bars2 = ax.barh(x + w/2, opdf['actual_risk_rate'], w, label='Actual Risk Rate (|δ|>0.5)',
                    color=COLORS['danger'], alpha=0.85, edgecolor='white')

    ax.set_yticks(x)
    ax.set_yticklabels(opdf['op_type'], fontsize=10)
    ax.set_xlabel('Rate (%)')
    ax.set_title('Comparison: Model Fallback Rate vs Actual Risk Rate')
    ax.legend(loc='lower right')

    # Annotate sample counts
    for i, row in enumerate(opdf.itertuples()):
        ax.text(max(row.fallback_rate, row.actual_risk_rate) + 1,
                i, f'n={row.n_total:,}', va='center', fontsize=8, color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_explain_risky_operators.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  FIGURE 8: Overall Confusion Matrix
# ===================================================================
def fig_confusion_matrix(df, results):
    """Overall 2x2 confusion matrix for high-risk detection."""
    print("\n  Generating: fig_explain_confusion_matrix.png")
    y = df['delta_time'].values
    ub = df['upper_bound'].values

    threshold = 0.5

    y_true = (np.abs(y) > threshold).astype(int)
    y_pred = (ub > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('HALO v4: Confusion Matrix — High-Risk Detection (|δ|>0.5)',
                 fontsize=15, fontweight='bold')

    # (a) Absolute counts
    ax = axes[0]
    labels = np.array([[f'TN\n{tn:,}\n({tn/total*100:.1f}%)',
                         f'FP\n{fp:,}\n({fp/total*100:.1f}%)'],
                        [f'FN\n{fn:,}\n({fn/total*100:.1f}%)',
                         f'TP\n{tp:,}\n({tp/total*100:.1f}%)']])
    colors_cm = np.array([[0.9, 0.2], [0.6, 0.1]])
    cmap = LinearSegmentedColormap.from_list('custom', ['#E8F5E9', '#2E7D32', '#FFEBEE', '#C62828'])
    im = ax.imshow([[tn, fp], [fn, tp]], cmap='RdYlGn_r', aspect='auto')

    for i in range(2):
        for j in range(2):
            color = 'white' if (i == j and i == 1) or (i != j and i == 1) else 'black'
            ax.text(j, i, labels[i, j], ha='center', va='center', fontsize=13, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Predicted SAFE', 'Predicted HIGH-RISK'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actual SAFE', 'Actual HIGH-RISK'])
    ax.set_title('(a) Confusion Matrix (Absolute)')

    # (b) Key metrics summary
    ax = axes[1]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    accuracy = (tp + tn) / total
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)

    metrics = {
        'Accuracy': accuracy,
        'Precision (PPV)': prec,
        'Recall (Sensitivity)': rec,
        'Specificity (TNR)': specificity,
        'NPV': npv,
        'F1 Score': f1,
    }

    names = list(metrics.keys())
    values = list(metrics.values())
    colors_bar = [COLORS['primary'] if v > 0.7 else COLORS['warning'] if v > 0.5 else COLORS['danger'] for v in values]
    bars = ax.barh(names, [v * 100 for v in values], color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Score (%)')
    ax.set_title('(b) Detection Metrics Summary')
    ax.set_xlim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_explain_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved")


# ===================================================================
#  MAIN
# ===================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("HALO v4 — Complete Explainability Figure Regeneration")
    print("=" * 60)

    # Step 1: Run CV
    df_pred, feature_cols, results, scaler = run_v4_cv()

    # Step 2: Generate all figures
    print("\n" + "=" * 60)
    print("STEP 2: Generating all explainability figures...")
    print("=" * 60)

    fig_sigma_evaluation(df_pred)
    fig_feature_importance(df_pred, feature_cols, scaler)
    fig_sigma_density(df_pred)
    fig_accuracy_by_operator(df_pred, results)
    fig_accuracy_by_env(df_pred, results)
    fig_decision_logic(df_pred, results)
    fig_risky_operators(df_pred, results)
    fig_confusion_matrix(df_pred, results)

    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO: {OUT_DIR}")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  1. fig_sigma_evaluation.png")
    print("  2. fig_sigma_feature_importance.png")
    print("  3. fig_sigma_density.png")
    print("  4. fig_explain_accuracy_operator.png")
    print("  5. fig_explain_accuracy_env.png")
    print("  6. fig_explain_decision_logic.png")
    print("  7. fig_explain_risky_operators.png")
    print("  8. fig_explain_confusion_matrix.png")
