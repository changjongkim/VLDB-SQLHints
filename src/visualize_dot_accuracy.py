# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def visualize_dot_accuracy():
    predictions_path = '/root/halo/results/sigma_predictions_v3.parquet'
    output_dir = '/root/halo/results/figures'
    
    if not os.path.exists(predictions_path):
        print("Data file not found.")
        return

    df = pd.read_parquet(predictions_path)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Plot 1: Predicted vs Actual Density (Hexbin) ──
    plt.figure(figsize=(11, 10))
    
    # Use hexbin to show density properly without "breaking" the plot
    hb = plt.hexbin(df['delta_time'], df['predicted_delta'], gridsize=50, 
                    cmap='YlGnBu', mincnt=1, bins='log', edgecolors='none', alpha=0.9)
    
    # Add identity line
    plt.plot([-4, 4], [-4, 4], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction (y=x)')
    
    # Add "High Risk" zones (Threshold 0.8)
    plt.axvline(0.8, color='#e74c3c', linestyle=':', alpha=0.6, linewidth=1.5)
    plt.axvline(-0.8, color='#e74c3c', linestyle=':', alpha=0.6, linewidth=1.5)
    plt.axhline(0.8, color='#e74c3c', linestyle=':', alpha=0.6, linewidth=1.5)
    plt.axhline(-0.8, color='#e74c3c', linestyle=':', alpha=0.6, linewidth=1.5)

    cb = plt.colorbar(hb, label='Log10(Count)')
    
    plt.title('Prediction Density: Actual vs. Predicted Operator Delta', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Delta (log Target/Source)', fontsize=14)
    plt.ylabel('Predicted Delta (log Target/Source)', fontsize=14)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma_density_accuracy.png'), dpi=300)
    plt.close()

    # ── Plot 2: Metrics Dot Plot (Cleaner than bars) ──
    THRESHOLD = 0.8
    df['is_high_risk'] = df['delta_time'].abs() > THRESHOLD
    df['pred_high_risk'] = df['predicted_delta'].abs() > THRESHOLD

    metrics_list = []
    for pair, group in df.groupby('pair_label'):
        y_true = group['is_high_risk']
        y_pred = group['pred_high_risk']
        
        pair_name = pair.replace('_vs_', ' \u2192 ')
        metrics_list.append({'Env Pair': pair_name, 'Score': precision_score(y_true, y_pred, zero_division=0), 'Metric': 'Precision'})
        metrics_list.append({'Env Pair': pair_name, 'Score': recall_score(y_true, y_pred, zero_division=0), 'Metric': 'Recall'})
        metrics_list.append({'Env Pair': pair_name, 'Score': f1_score(y_true, y_pred, zero_division=0), 'Metric': 'F1-Score'})
    
    df_metrics = pd.DataFrame(metrics_list)

    plt.figure(figsize=(12, 6))
    # Pointplot creates the "dot" style with lines connecting metrics for each env
    sns.pointplot(data=df_metrics, x='Env Pair', y='Score', hue='Metric', 
                  join=False, palette='bright', markers=["o", "s", "D"], scale=1.2)
    
    plt.title(f'Classification Metrics Dot Plot (Threshold: {THRESHOLD})', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_metrics_dotplot.png'), dpi=300)
    plt.close()

    print("Generated dot-style accuracy plots in /root/halo/results/figures/")

if __name__ == "__main__":
    visualize_dot_accuracy()
