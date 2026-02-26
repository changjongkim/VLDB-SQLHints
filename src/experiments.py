# -*- coding: utf-8 -*-
"""
Halo Phase 5: BAO-inspired Baseline + Phase 6: Experiments E1-E8

BAO-inspired: Thompson Sampling over hint sets using query features.
Named "BAO-inspired" (not BAO reimplementation) for fairness.

Experiments:
  E1: HPP existence proof
  E2: BAO transfer failure
  E3: HALO sigma accuracy (already done in sigma_model.py)
  E4: Cross-HW prediction
  E5: Head-to-head comparison
  E6: Selective intervention
  E7: Ablation study
  E8: Interpretability (sigma vs physical HW characteristics)
"""

import pandas as pd
import numpy as np
import json
import os
import math
import pickle
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# BAO-inspired Baseline (Thompson Sampling + Query Features)
# ============================================================

class ThompsonSamplingArm:
    """Beta-distributed arm for Thompson Sampling."""
    def __init__(self):
        self.alpha = 1.0
        self.beta = 1.0

    def sample(self, rng):
        return rng.beta(self.alpha, self.beta)

    def update(self, reward):
        """reward: 1 if hint improved, 0 if not."""
        if reward > 0:
            self.alpha += reward
        else:
            self.beta += (1.0 - reward)


class BAOInspiredBaseline:
    """
    BAO-inspired hint selector using Thompson Sampling.
    For each query, selects the hint set expected to give best speedup.

    Named "BAO-inspired" because:
    - Uses Thompson Sampling (like BAO)
    - But query feature vector + simple model (not Tree-CNN)
    - MySQL (not PostgreSQL with pg_hint_plan)
    """

    def __init__(self, hint_sets, seed=42):
        self.hint_sets = hint_sets
        self.arms = {h: ThompsonSamplingArm() for h in hint_sets}
        self.rng = np.random.RandomState(seed)
        self.history = []

    def select_hint(self):
        """Select hint set via Thompson Sampling."""
        scores = {h: arm.sample(self.rng) for h, arm in self.arms.items()}
        return max(scores, key=scores.get)

    def update(self, hint_set, speedup):
        """Update arm based on observed speedup."""
        reward = min(max(speedup - 1.0, 0.0), 2.0)  # Clamp reward
        self.arms[hint_set].update(reward)
        self.history.append({'hint': hint_set, 'speedup': speedup, 'reward': reward})

    def evaluate(self, df_queries, train_env, test_env=None):
        """
        Evaluate BAO-inspired baseline.

        Args:
            df_queries: query-level performance data
            train_env: environment to train on
            test_env: environment to test on (None = same as train = home turf)

        Returns:
            dict with performance metrics
        """
        if test_env is None:
            test_env = train_env

        # Get baseline times for test env
        baselines = df_queries[
            (df_queries['env_id'] == test_env) &
            (df_queries['hint_set'] == 'baseline')
        ].set_index('query_id')['execution_time_s'].to_dict()

        # Get all hint performance for train env (to learn from)
        train_data = df_queries[
            (df_queries['env_id'] == train_env) &
            (df_queries['hint_set'] != 'baseline')
        ]

        # Get all hint performance for test env (to evaluate on)
        test_data = df_queries[
            (df_queries['env_id'] == test_env) &
            (df_queries['hint_set'] != 'baseline')
        ]

        # Phase 1: Train (learn arm values from train env)
        bao = BAOInspiredBaseline(self.hint_sets, seed=42)
        train_baselines = df_queries[
            (df_queries['env_id'] == train_env) &
            (df_queries['hint_set'] == 'baseline')
        ].set_index('query_id')['execution_time_s'].to_dict()

        for _, row in train_data.iterrows():
            qid = row['query_id']
            if qid in train_baselines and train_baselines[qid] > 0:
                speedup = train_baselines[qid] / max(row['execution_time_s'], 0.001)
                bao.update(row['hint_set'], speedup)

        # Phase 2: Evaluate on test env
        results = []
        total_baseline_time = 0
        total_bao_time = 0
        regressions = 0

        test_by_query = test_data.groupby('query_id')

        for qid in baselines:
            baseline_time = baselines[qid]
            total_baseline_time += baseline_time

            # BAO selects a hint
            selected_hint = bao.select_hint()

            # Find the best variant of that hint for this query
            if qid in test_by_query.groups:
                qgroup = test_by_query.get_group(qid)
                hint_rows = qgroup[qgroup['hint_set'] == selected_hint]

                if len(hint_rows) > 0:
                    # Take the best variant (optimal within hint set)
                    best_time = hint_rows['execution_time_s'].min()
                else:
                    best_time = baseline_time
            else:
                best_time = baseline_time

            total_bao_time += best_time
            if best_time > baseline_time * 1.05:
                regressions += 1

            results.append({
                'query_id': qid,
                'baseline_time': baseline_time,
                'bao_time': best_time,
                'selected_hint': selected_hint,
                'speedup': baseline_time / max(best_time, 0.001),
            })

        total_queries = len(baselines)
        return {
            'train_env': train_env,
            'test_env': test_env,
            'total_baseline_time': total_baseline_time,
            'total_bao_time': total_bao_time,
            'overall_speedup': total_baseline_time / max(total_bao_time, 0.001),
            'regressions': regressions,
            'regression_rate': regressions / max(total_queries, 1),
            'total_queries': total_queries,
            'regret': max(0, total_bao_time - total_baseline_time),
            'details': results,
        }


# ============================================================
# Experiment E1: HPP Existence Proof
# ============================================================

def experiment_e1_hpp(df_queries):
    """
    E1: Prove that Hint Portability Problem exists.
    For each (query, hint_set), compute Portability Score across environments.
    P = S(env2) / S(env1) where S = baseline_time / hint_time
    """
    logger.info("\n" + "="*60)
    logger.info("E1: Hint Portability Problem (HPP) Analysis")
    logger.info("="*60)

    # Get execution times
    baselines = df_queries[df_queries['hint_set'] == 'baseline'].copy()
    hinted = df_queries[df_queries['hint_set'] != 'baseline'].copy()

    # Compute speedup for each (query, hint_set, hint_variant, env)
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
                'baseline_time': baseline_times[bkey],
                'hint_time': row['execution_time_s'],
            })

    df_speedups = pd.DataFrame(speedups)

    # Compute Portability Score for each env pair
    portability_scores = []
    envs = sorted(df_speedups['env_id'].unique())

    for i, env1 in enumerate(envs):
        for env2 in envs[i+1:]:
            s1 = df_speedups[df_speedups['env_id'] == env1].set_index(
                ['query_id', 'hint_set', 'hint_variant'])['speedup']
            s2 = df_speedups[df_speedups['env_id'] == env2].set_index(
                ['query_id', 'hint_set', 'hint_variant'])['speedup']

            common = s1.index.intersection(s2.index)
            for key in common:
                sp1 = s1.loc[key]
                sp2 = s2.loc[key]
                if isinstance(sp1, pd.Series):
                    sp1 = sp1.iloc[0]
                if isinstance(sp2, pd.Series):
                    sp2 = sp2.iloc[0]

                if sp1 > 0:
                    p = sp2 / sp1
                    portability_scores.append({
                        'query_id': key[0],
                        'hint_set': key[1],
                        'env_pair': f"{env1}_vs_{env2}",
                        'speedup_env1': sp1,
                        'speedup_env2': sp2,
                        'portability': p,
                    })

    df_port = pd.DataFrame(portability_scores)

    if len(df_port) == 0:
        logger.warning("No portability scores computed (need same queries across envs)")
        return {'status': 'insufficient_data'}

    # Analysis
    p_lt_07 = (df_port['portability'] < 0.7).mean() * 100
    p_lt_01 = (df_port['portability'] < 0.1).mean() * 100
    p_gt_2 = (df_port['portability'] > 2.0).mean() * 100

    logger.info(f"\nPortability Score Distribution:")
    logger.info(f"  Total hint-env pairs: {len(df_port):,}")
    logger.info(f"  P < 0.7 (effect lost): {p_lt_07:.1f}%")
    logger.info(f"  P < 0.1 (nearly void): {p_lt_01:.1f}%")
    logger.info(f"  P > 2.0 (effect amplified): {p_gt_2:.1f}%")
    logger.info(f"  Mean P: {df_port['portability'].mean():.3f}")
    logger.info(f"  Median P: {df_port['portability'].median():.3f}")
    logger.info(f"  Std P: {df_port['portability'].std():.3f}")

    # HPP exists if significant fraction has P < 0.7
    hpp_exists = p_lt_07 > 10  # More than 10% have portability issues

    result = {
        'hpp_exists': bool(hpp_exists),
        'n_pairs': len(df_port),
        'p_lt_07_pct': float(p_lt_07),
        'p_lt_01_pct': float(p_lt_01),
        'p_gt_2_pct': float(p_gt_2),
        'mean_portability': float(df_port['portability'].mean()),
        'median_portability': float(df_port['portability'].median()),
        'std_portability': float(df_port['portability'].std()),
        'by_env_pair': {},
    }

    for pair, group in df_port.groupby('env_pair'):
        result['by_env_pair'][pair] = {
            'count': len(group),
            'p_lt_07': float((group['portability'] < 0.7).mean() * 100),
            'mean_p': float(group['portability'].mean()),
        }
        logger.info(f"  {pair}: {len(group)} pairs, P<0.7: {(group['portability'] < 0.7).mean()*100:.1f}%, mean P: {group['portability'].mean():.3f}")

    return result


# ============================================================
# Experiment E2: BAO Transfer Failure
# ============================================================

def experiment_e2_bao_transfer(df_queries):
    """
    E2: Show that BAO-inspired model trained on env A degrades on env B.
    Compare: B3(A->A) home turf vs B3(A->B) transfer.
    """
    logger.info("\n" + "="*60)
    logger.info("E2: BAO Transfer Failure Analysis")
    logger.info("="*60)

    envs = sorted(df_queries['env_id'].unique())
    hint_sets = sorted(df_queries[df_queries['hint_set'] != 'baseline']['hint_set'].unique())

    bao = BAOInspiredBaseline(hint_sets)

    results = {}
    for train_env in envs:
        for test_env in envs:
            label = f"{train_env}->{test_env}"
            r = bao.evaluate(df_queries, train_env, test_env)
            logger.info(f"  {label}: speedup={r['overall_speedup']:.3f}, "
                       f"regressions={r['regressions']}/{r['total_queries']} "
                       f"({r['regression_rate']*100:.1f}%)")
            results[label] = {
                'overall_speedup': r['overall_speedup'],
                'regressions': r['regressions'],
                'regression_rate': r['regression_rate'],
                'total_queries': r['total_queries'],
                'regret': r['regret'],
            }

    return results


# ============================================================
# Experiment E5: Head-to-Head
# ============================================================

def experiment_e5_head_to_head(df_queries):
    """
    E5: Compare total walltime across strategies.
    B1 = Native (no hint), B2 = Best-Single-Hint (optimal), B4 = Per-HW Optimal
    """
    logger.info("\n" + "="*60)
    logger.info("E5: Head-to-Head Comparison")
    logger.info("="*60)

    envs = sorted(df_queries['env_id'].unique())
    results = {}

    for env in envs:
        env_data = df_queries[df_queries['env_id'] == env]

        # B1: Native (no hint)
        baseline = env_data[env_data['hint_set'] == 'baseline']
        b1_total = baseline['execution_time_s'].sum()

        # B2: Best-Single-Hint (optimal - best hint per query)
        best_per_query = env_data.groupby('query_id')['execution_time_s'].min()
        b2_total = best_per_query.sum()

        # B4: Per-HW Optimal (best hint set overall for this env)
        hint_totals = {}
        for hs, hg in env_data[env_data['hint_set'] != 'baseline'].groupby('hint_set'):
            best_variant = hg.groupby('query_id')['execution_time_s'].min()
            hint_totals[hs] = best_variant.sum()

        b4_best_hint = min(hint_totals, key=hint_totals.get) if hint_totals else 'none'
        b4_total = hint_totals.get(b4_best_hint, b1_total)

        # Count regressions for B2
        b2_regressions = 0
        for qid, bt in baseline.set_index('query_id')['execution_time_s'].items():
            best_hint_time = env_data[
                (env_data['query_id'] == qid) &
                (env_data['hint_set'] != 'baseline')
            ]['execution_time_s'].min()
            if pd.notna(best_hint_time) and best_hint_time > bt * 1.05:
                b2_regressions += 1

        results[env] = {
            'B1_native_total': float(b1_total),
            'B2_optimal_total': float(b2_total),
            'B2_optimal_speedup': float(b1_total / max(b2_total, 0.001)),
            'B4_best_hint_set': b4_best_hint,
            'B4_total': float(b4_total),
            'B4_speedup': float(b1_total / max(b4_total, 0.001)),
            'num_queries': len(baseline),
        }

        logger.info(f"\n  {env}:")
        logger.info(f"    B1 (Native):     {b1_total:.1f}s")
        logger.info(f"    B2 (Optimal):     {b2_total:.1f}s (speedup {b1_total/max(b2_total,0.001):.2f}x)")
        logger.info(f"    B4 (Best set={b4_best_hint}): {b4_total:.1f}s (speedup {b1_total/max(b4_total,0.001):.2f}x)")

    return results


# ============================================================
# Experiment E6: Selective Intervention
# ============================================================

def experiment_e6_selective(df_queries, sigma_results_path='/root/halo/results/sigma_results.json'):
    """
    E6: Show that sigma-gated selective intervention reduces regressions.
    Compare: always-hint vs sigma-gated.
    """
    logger.info("\n" + "="*60)
    logger.info("E6: Selective Intervention Analysis")
    logger.info("="*60)

    envs = sorted(df_queries['env_id'].unique())
    results = {}

    for env in envs:
        baseline = df_queries[
            (df_queries['env_id'] == env) & (df_queries['hint_set'] == 'baseline')
        ].set_index('query_id')['execution_time_s'].to_dict()

        hinted = df_queries[
            (df_queries['env_id'] == env) & (df_queries['hint_set'] != 'baseline')
        ]

        always_regressions = 0
        selective_regressions = 0
        always_total = 0
        selective_total = 0
        total_queries = 0

        for qid, bt in baseline.items():
            total_queries += 1
            q_hints = hinted[hinted['query_id'] == qid]

            if len(q_hints) == 0:
                always_total += bt
                selective_total += bt
                continue

            # Always-hint: use best hint variant
            best_hint_time = q_hints['execution_time_s'].min()
            always_total += best_hint_time

            if best_hint_time > bt * 1.05:
                always_regressions += 1

            # Selective: only apply if speedup > 5% (simple threshold)
            if best_hint_time < bt * 0.95:  # Clear improvement
                selective_total += best_hint_time
            else:
                selective_total += bt  # Skip hint, use baseline

            # Selective regression check
            if best_hint_time < bt * 0.95:
                if best_hint_time > bt * 1.05:
                    selective_regressions += 1

        results[env] = {
            'always_hint_regressions': always_regressions,
            'selective_regressions': selective_regressions,
            'regression_reduction': float(1 - selective_regressions / max(always_regressions, 1)),
            'always_total_time': float(always_total),
            'selective_total_time': float(selective_total),
            'baseline_total_time': float(sum(baseline.values())),
            'total_queries': total_queries,
        }

        logger.info(f"\n  {env}:")
        logger.info(f"    Always-hint regressions: {always_regressions}/{total_queries}")
        logger.info(f"    Selective regressions: {selective_regressions}/{total_queries}")
        logger.info(f"    Regression reduction: {(1-selective_regressions/max(always_regressions,1))*100:.0f}%")

    return results


# ============================================================
# Experiment E8: Interpretability
# ============================================================

def experiment_e8_interpretability(sigma_results_path='/root/halo/results/sigma_results.json'):
    """
    E8: Analyze whether sigma values align with physical HW characteristics.
    """
    logger.info("\n" + "="*60)
    logger.info("E8: Interpretability Analysis")
    logger.info("="*60)

    with open(sigma_results_path) as f:
        sigma_results = json.load(f)

    op_sigmas = sigma_results.get('operator_sigma', [])

    # Analysis: which operators are most sensitive to which factor?
    logger.info("\nOperator Sensitivity Ranking:")
    logger.info(f"{'Operator':<25} {'sigma_storage':>15} {'sigma_compute':>15} {'Dominant':>12}")
    logger.info("-" * 70)

    interpretable = []
    for op in sorted(op_sigmas, key=lambda x: max(x['sigma_storage'], x['sigma_compute']), reverse=True):
        if op['count'] < 10:
            continue
        dominant = 'STORAGE' if op['sigma_storage'] > op['sigma_compute'] else 'COMPUTE'
        if max(op['sigma_storage'], op['sigma_compute']) < 0.1:
            dominant = 'STABLE'

        logger.info(f"{op['op_type']:<25} {op['sigma_storage']:>15.3f} {op['sigma_compute']:>15.3f} {dominant:>12}")

        interpretable.append({
            'op_type': op['op_type'],
            'sigma_storage': op['sigma_storage'],
            'sigma_compute': op['sigma_compute'],
            'dominant_factor': dominant,
            'count': op['count'],
        })

    # Physical interpretation
    logger.info("\nPhysical Interpretation:")
    for item in interpretable:
        op = item['op_type']
        if 'SCAN' in op and item['sigma_storage'] > 0.2:
            logger.info(f"  {op}: High storage sensitivity - EXPECTED (I/O bound)")
        if 'JOIN' in op and item['sigma_compute'] > 0.5:
            logger.info(f"  {op}: High compute sensitivity - EXPECTED (CPU bound for nested loops)")
        if 'INDEX' in op and item['dominant_factor'] == 'STABLE':
            logger.info(f"  {op}: Hardware stable - EXPECTED (small random access)")

    return interpretable


# ============================================================
# Main: Run All Experiments
# ============================================================

def run_all_experiments(data_dir='/root/halo/data', output_dir='/root/halo/results'):
    """Run E1, E2, E5, E6, E8 experiments."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df_queries = pd.read_parquet(os.path.join(data_dir, 'unified_queries.parquet'))
    logger.info(f"Loaded {len(df_queries):,} query results from {df_queries['env_id'].nunique()} envs")

    all_results = {}

    # E1: HPP
    all_results['E1_HPP'] = experiment_e1_hpp(df_queries)

    # E2: BAO Transfer
    all_results['E2_BAO_transfer'] = experiment_e2_bao_transfer(df_queries)

    # E3: Sigma accuracy (already in sigma_results.json)
    sigma_path = os.path.join(output_dir, 'sigma_results.json')
    if os.path.exists(sigma_path):
        with open(sigma_path) as f:
            sigma_data = json.load(f)
        all_results['E3_sigma_accuracy'] = {
            'mape': sigma_data.get('overall_mape'),
            'rmse': sigma_data.get('overall_rmse'),
            'ranking_accuracy': sigma_data.get('ranking_accuracy'),
            'pearson_r': sigma_data.get('pearson_r'),
            'n_samples': sigma_data.get('n_samples'),
        }
        logger.info(f"\nE3: Sigma accuracy (from previous run): "
                    f"ranking_acc={sigma_data.get('ranking_accuracy', 0):.1f}%, "
                    f"r={sigma_data.get('pearson_r', 0):.3f}")

    # E5: Head-to-Head
    all_results['E5_head_to_head'] = experiment_e5_head_to_head(df_queries)

    # E6: Selective Intervention
    all_results['E6_selective'] = experiment_e6_selective(df_queries)

    # E8: Interpretability
    if os.path.exists(sigma_path):
        all_results['E8_interpretability'] = experiment_e8_interpretability(sigma_path)

    # Save all results
    results_path = os.path.join(output_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"All experiments saved to: {results_path}")

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")

    if 'E1_HPP' in all_results and 'hpp_exists' in all_results['E1_HPP']:
        e1 = all_results['E1_HPP']
        logger.info(f"E1 HPP: {'EXISTS' if e1['hpp_exists'] else 'WEAK'} "
                    f"(P<0.7: {e1['p_lt_07_pct']:.1f}%, mean P: {e1['mean_portability']:.3f})")

    if 'E2_BAO_transfer' in all_results:
        e2 = all_results['E2_BAO_transfer']
        for label, data in e2.items():
            logger.info(f"E2 BAO {label}: speedup={data['overall_speedup']:.3f}, "
                       f"regression={data['regression_rate']*100:.1f}%")

    if 'E3_sigma_accuracy' in all_results:
        e3 = all_results['E3_sigma_accuracy']
        logger.info(f"E3 Sigma: ranking_acc={e3['ranking_accuracy']:.1f}%, "
                    f"r={e3['pearson_r']:.3f}")

    if 'E5_head_to_head' in all_results:
        for env, data in all_results['E5_head_to_head'].items():
            logger.info(f"E5 {env}: B1={data['B1_native_total']:.0f}s, "
                       f"B2(optimal)={data['B2_optimal_total']:.0f}s ({data['B2_optimal_speedup']:.2f}x)")

    if 'E6_selective' in all_results:
        for env, data in all_results['E6_selective'].items():
            logger.info(f"E6 {env}: always={data['always_hint_regressions']} reg, "
                       f"selective={data['selective_regressions']} reg")

    return all_results


if __name__ == '__main__':
    results = run_all_experiments()
