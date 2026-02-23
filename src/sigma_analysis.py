# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def analyze_sigma_classification():
    with open('/root/halo/results/sigma_results_v2.json', 'r') as f:
        data = json.json_load(f)
    
    # In a real scenario, we would need the raw predictions. 
    # Since we only have the overall metrics in the JSON, let's focus on 
    # summarizing what we found in the previous step and adding more insights 
    # if we can derive them.
    # Actually, let's recalculate metrics if we had the test data.
    # Since I don't have the test data easily accessible here without re-running training code,
    # I will create a script that runs the evaluation on the saved model to get these metrics.
    
    print("=== σ Model Deep Analysis (Classification Focus) ===")
    print(f"Overall R2: {data['overall_r2']:.4f}")
    print(f"Overall Direction Accuracy: {data['direction_accuracy']:.2f}%")
    
    # Logic for High-Risk Classification Accuracy
    # To do this properly, I'd need to run the model on the full operator dataset.
    
if __name__ == "__main__":
    # Instead of just printing, let's actually perform the analysis by loading the model and data.
    import pickle
    import xgboost as xgb
    
    print("Loading model and data for deep analysis...")
    with open('/root/halo/results/sigma_model_v2.pkl', 'rb') as f:
        saved = pickle.load(f)
        model = saved['model']
        feature_cols = saved['feature_cols']
        
    df_samples = pd.read_parquet('/root/halo/data/unified_operators.parquet') # This is raw data, we need the sample pairs.
    # We need to reconstruct the pairs to evaluate.
    # For now, let's use the summary we have and prepare the logic to run in a more integrated script.
    
    print("Analysis script prepared. Will be integrated into sigma_model_v2.py for direct metric extraction.")
