# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def visualize_accuracy_metrics():
    results_path = '/root/halo/results/sigma_results_v3.json'
    predictions_path = '/root/halo/results/sigma_predictions_v3.parquet'
    output_dir = '/root/halo/results/figures'
    
    if not os.path.exists(results_path) or not os.path.exists(predictions_path):
        print("Data files not found.")
        return

    with open(results_path, 'r') as f:
        res = json.load(f)
    df = pd.read_parquet(predictions_path)

    # Classification Threshold
    THRESHOLD = 0.8
    df['is_high_risk'] = df['delta_time'].abs() > THRESHOLD
    df['pred_high_risk'] = df['predicted_delta'].abs() > THRESHOLD

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Plot 1: Classification Metrics by Environment Pair ──
    metrics_list = []
    for pair, group in df.groupby('pair_label'):
        y_true = group['is_high_risk']
        y_pred = group['pred_high_risk']
        
        metrics_list.append({
            'Env Pair': pair.replace('_vs_', ' \u2192 '),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_true, y_pred)
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics_melted = df_metrics.melt(id_vars='Env Pair', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_metrics_melted, x='Env Pair', y='Score', hue='Metric', palette='muted')
    plt.title(f'Classification Metrics for High-Risk Detection (Threshold: |Δ| > {THRESHOLD})', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_by_env.png'), dpi=300)
    plt.close()

    # ── Plot 2: Metrics by Operator Type (Top 10 most frequent) ──
    op_metrics = []
    top_ops = df['op_type'].value_counts().head(10).index
    for op in top_ops:
        group = df[df['op_type'] == op]
        y_true = group['is_high_risk']
        y_pred = group['pred_high_risk']
        
        if y_true.sum() > 0: # Only include if there's at least one high-risk case
            op_metrics.append({
                'Operator': op,
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, zero_division=0)
            })
    
    df_op_metrics = pd.DataFrame(op_metrics)
    df_op_melted = df_op_metrics.melt(id_vars='Operator', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_op_melted, x='Operator', y='Score', hue='Metric', palette='viridis')
    plt.title('High-Risk Detection Metrics by Operator Type', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_by_operator.png'), dpi=300)
    plt.close()

    # ── Plot 3: Overall Confusion Matrix ──
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df['is_high_risk'], df['pred_high_risk'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'High Risk'], 
                yticklabels=['Safe', 'High Risk'],
                annot_kws={"size": 16})
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Predicted', fontsize=13)
    plt.title('Overall Confusion Matrix (All Envs)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_confusion_matrix.png'), dpi=300)
    plt.close()

    print("Generated accuracy metric plots in /root/halo/results/figures/")

if __name__ == "__main__":
    visualize_accuracy_metrics()
