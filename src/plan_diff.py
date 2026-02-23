# -*- coding: utf-8 -*-
"""
Halo Phase 2: Plan Diff Extractor

Identifies which operators changed between baseline and hinted executions
for the same query in the same environment.

Key insight: Since these are the same query with/without hints,
table alias-based matching is sufficient (no full tree edit distance needed).

Changes detected:
  1. join_method_change: NL -> Hash, etc.
  2. access_method_change: TableScan -> IndexLookup
  3. operator_added/removed
  4. timing_change: same structure but different performance
"""

import pandas as pd
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_dir='/root/halo/data'):
    """Load unified parquet files."""
    df_ops = pd.read_parquet(os.path.join(data_dir, 'unified_operators.parquet'))
    df_queries = pd.read_parquet(os.path.join(data_dir, 'unified_queries.parquet'))
    return df_ops, df_queries


def extract_plan_diffs(df_ops, df_queries):
    """
    For each (query_id, env_id, hint_set) where hint_set != baseline,
    find the matching baseline and compute operator-level differences.

    Returns a DataFrame of operator-level diffs with columns:
      - query_id, env_id, hint_set, hint_variant
      - op_type_baseline, op_type_hinted
      - change_type: 'same', 'method_change', 'added', 'removed', 'timing_only'
      - self_time_baseline, self_time_hinted
      - time_ratio: hinted / baseline
      - all operator features for both baseline and hinted
    """
    diffs = []

    # Get baseline operators grouped by (query_id, env_id)
    baseline_ops = df_ops[df_ops['hint_set'] == 'baseline'].copy()
    hinted_ops = df_ops[df_ops['hint_set'] != 'baseline'].copy()

    # Group baselines by (query_id, env_id) for fast lookup
    baseline_groups = {}
    for key, group in baseline_ops.groupby(['query_id', 'env_id']):
        baseline_groups[key] = group

    # Process each hinted execution
    for (query_id, env_id, hint_set, hint_variant), hint_group in \
            hinted_ops.groupby(['query_id', 'env_id', 'hint_set', 'hint_variant']):

        baseline_key = (query_id, env_id)
        if baseline_key not in baseline_groups:
            continue

        base_group = baseline_groups[baseline_key]

        # Match operators by tree_position (DFS order)
        # This works because same query => similar tree structure
        base_by_pos = {row['tree_position']: row for _, row in base_group.iterrows()}
        hint_by_pos = {row['tree_position']: row for _, row in hint_group.iterrows()}

        all_positions = set(base_by_pos.keys()) | set(hint_by_pos.keys())

        for pos in sorted(all_positions):
            base_op = base_by_pos.get(pos)
            hint_op = hint_by_pos.get(pos)

            if base_op is not None and hint_op is not None:
                # Both exist - check if operator type changed
                if base_op['op_type'] != hint_op['op_type']:
                    change_type = 'method_change'
                else:
                    change_type = 'timing_only'

                base_self = max(base_op['self_time'], 0.001)
                hint_self = max(hint_op['self_time'], 0.001)

                diffs.append({
                    'query_id': query_id,
                    'env_id': env_id,
                    'benchmark': base_op['benchmark'],
                    'hint_set': hint_set,
                    'hint_variant': hint_variant,
                    'tree_position': pos,
                    'change_type': change_type,
                    # Baseline features
                    'op_type_baseline': base_op['op_type'],
                    'table_alias_baseline': base_op['table_alias'],
                    'est_cost_baseline': base_op['est_cost'],
                    'est_rows_baseline': base_op['est_rows'],
                    'actual_rows_baseline': base_op['actual_rows'],
                    'loops_baseline': base_op['loops'],
                    'self_time_baseline': base_op['self_time'],
                    'total_work_baseline': base_op['total_work'],
                    'row_est_error_baseline': base_op['row_est_error'],
                    'depth_baseline': base_op['depth'],
                    'parent_op_type_baseline': base_op['parent_op_type'],
                    # Hinted features
                    'op_type_hinted': hint_op['op_type'],
                    'table_alias_hinted': hint_op['table_alias'],
                    'est_cost_hinted': hint_op['est_cost'],
                    'est_rows_hinted': hint_op['est_rows'],
                    'actual_rows_hinted': hint_op['actual_rows'],
                    'loops_hinted': hint_op['loops'],
                    'self_time_hinted': hint_op['self_time'],
                    'total_work_hinted': hint_op['total_work'],
                    'row_est_error_hinted': hint_op['row_est_error'],
                    'depth_hinted': hint_op['depth'],
                    'parent_op_type_hinted': hint_op['parent_op_type'],
                    # Derived
                    'time_ratio': hint_self / base_self,
                    'time_diff_ms': hint_op['self_time'] - base_op['self_time'],
                })

            elif base_op is not None:
                diffs.append({
                    'query_id': query_id,
                    'env_id': env_id,
                    'benchmark': base_op['benchmark'],
                    'hint_set': hint_set,
                    'hint_variant': hint_variant,
                    'tree_position': pos,
                    'change_type': 'removed',
                    'op_type_baseline': base_op['op_type'],
                    'table_alias_baseline': base_op['table_alias'],
                    'self_time_baseline': base_op['self_time'],
                    'op_type_hinted': 'NONE',
                    'self_time_hinted': 0.0,
                    'time_ratio': 0.0,
                    'time_diff_ms': -base_op['self_time'],
                })
            else:
                diffs.append({
                    'query_id': query_id,
                    'env_id': env_id,
                    'benchmark': hint_op['benchmark'],
                    'hint_set': hint_set,
                    'hint_variant': hint_variant,
                    'tree_position': pos,
                    'change_type': 'added',
                    'op_type_baseline': 'NONE',
                    'op_type_hinted': hint_op['op_type'],
                    'table_alias_hinted': hint_op['table_alias'],
                    'self_time_baseline': 0.0,
                    'self_time_hinted': hint_op['self_time'],
                    'time_ratio': float('inf'),
                    'time_diff_ms': hint_op['self_time'],
                })

    df_diffs = pd.DataFrame(diffs)
    return df_diffs


def run_plan_diff(data_dir='/root/halo/data', output_dir='/root/halo/data'):
    """Run Plan Diff extraction and save results."""
    logger.info("Loading data...")
    df_ops, df_queries = load_data(data_dir)

    logger.info(f"Operators: {len(df_ops):,}, Queries: {len(df_queries):,}")
    logger.info("Extracting plan diffs...")

    df_diffs = extract_plan_diffs(df_ops, df_queries)

    # Save
    diffs_path = os.path.join(output_dir, 'plan_diffs.parquet')
    df_diffs.to_parquet(diffs_path, index=False)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Plan Diff Complete!")
    logger.info(f"  Total operator diffs: {len(df_diffs):,}")
    logger.info(f"  Change types:")
    for ct, count in df_diffs['change_type'].value_counts().items():
        logger.info(f"    {ct}: {count:,}")
    logger.info(f"  Environments: {df_diffs['env_id'].nunique()}")
    logger.info(f"  Queries with diffs: {df_diffs['query_id'].nunique()}")
    logger.info(f"  Saved: {diffs_path}")

    # Save summary
    summary = {
        'total_diffs': len(df_diffs),
        'change_types': df_diffs['change_type'].value_counts().to_dict(),
        'envs': sorted(df_diffs['env_id'].unique().tolist()),
        'queries_with_diffs': df_diffs['query_id'].nunique(),
        'method_changes': int((df_diffs['change_type'] == 'method_change').sum()),
    }
    with open(os.path.join(output_dir, 'plan_diff_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return df_diffs, summary


if __name__ == '__main__':
    df_diffs, summary = run_plan_diff()
    print(json.dumps(summary, indent=2))
