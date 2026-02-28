# -*- coding: utf-8 -*-
"""
HALO v4 — Phase 2: Lightweight Bottom-Up Message Passing
=========================================================

Adds tree-structural context to flat tabular operator features WITHOUT
requiring expensive GNN/Tree-LSTM architectures.

Problem (v3):
  Each operator is extracted as an independent row (flat tabular data).
  A child TABLE_SCAN with I/O stall forces the parent HASH_JOIN to spin-wait,
  but the parent's features know NOTHING about this downstream bottleneck.
  Competing work (BAO, Neo) uses Tree-LSTM/GNN to capture this, making
  the v3 flat-tabular approach look naive.

Solution (v4):
  Post-order traversal (leaves → root) propagates aggregated child features
  up to parent nodes. This injects structural context into the flat features
  that Random Forest / MLP can consume.

Added Features per operator:
  1. child_max_io_bottleneck:  max I/O risk score among all descendants
  2. subtree_cumulative_cost:  total est_cost accumulated from entire subtree
  3. subtree_total_work:       total work (self_time * loops) from subtree
  4. subtree_max_depth:        deepest leaf below this node
  5. is_pipeline_breaker:      1 if this node blocks on a slow child
  6. child_selectivity_risk:   max cardinality estimation error in subtree

Academic Defense:
  "Rather than adopting heavyweight GNN or Tree-LSTM architectures that
   introduce significant inference latency, HALO employs a lightweight
   post-order feature aggregation scheme. This single-pass O(n) traversal
   propagates critical bottleneck signals (I/O stalls, cardinality
   explosions) from leaf to root, enabling the tabular model to capture
   inter-operator dependencies without changing its core architecture."
"""

import math
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  I/O Risk Scoring
# ═══════════════════════════════════════════════════════════════════════

# Operator types that are I/O-intensive
IO_INTENSIVE_OPS = {
    'TABLE_SCAN', 'INDEX_SCAN', 'INDEX_LOOKUP',
    'INDEX_RANGE_SCAN', 'TEMP_TABLE_SCAN'
}

# Operator types that are blocking (must consume ALL child output
# before producing any output — pipeline breakers)
BLOCKING_OPS = {
    'SORT', 'HASH_JOIN', 'AGGREGATE', 'GROUP_AGGREGATE',
    'MATERIALIZE', 'HASH_BUILD'
}


def compute_io_risk_score(op: dict) -> float:
    """
    Compute an I/O risk score for a single operator.
    
    Higher score = this operator is more likely to cause I/O bottlenecks.
    Combines:
      - Whether the op is I/O-intensive (scan, index lookup)
      - How much data it processes (total_work)
      - Whether it does random vs sequential access (loops as proxy)
    
    Returns:
        float in [0, ∞), where 0 = no I/O risk
    """
    op_type = op.get('op_type', '')
    
    if op_type not in IO_INTENSIVE_OPS:
        return 0.0
    
    total_work = max(op.get('total_work', 0.001), 0.001)
    loops = max(op.get('loops', 1), 1)
    actual_rows = max(op.get('actual_rows', 1), 1)
    
    # Base I/O intensity from work volume
    io_base = math.log(total_work + 1)
    
    # Random access penalty: high loops = repeated random seeks
    # Sequential scans (loops=1) are much cheaper than looped lookups
    random_access_penalty = math.log(loops + 1) * 0.5
    
    # Row volume factor
    row_factor = math.log(actual_rows + 1) * 0.3
    
    return io_base + random_access_penalty + row_factor


def compute_selectivity_risk(op: dict) -> float:
    """
    Compute cardinality estimation risk for a single operator.
    
    Large Q-errors (row_est_error) indicate the optimizer's cost model
    is wrong for this operator → hints based on faulty cost estimates
    are more likely to regress.
    """
    q_error = abs(op.get('row_est_error', 0.0))
    return q_error


# ═══════════════════════════════════════════════════════════════════════
#  Post-Order Traversal Aggregator
# ═══════════════════════════════════════════════════════════════════════

def propagate_tree_features(operators: List[dict]) -> List[dict]:
    """
    Bottom-up (post-order) feature propagation through the operator tree.
    
    For each operator, computes aggregated subtree features by traversing
    from leaves to root. This is a single-pass O(n) algorithm.
    
    Args:
        operators: list of operator dicts (from parse_tree/parquet),
                   must have 'children_indices', 'parent_idx', 'tree_position'
                   
    Returns:
        Same list with new structural features added to each operator dict.
    
    New features added:
        child_max_io_bottleneck:  max I/O risk in entire subtree
        subtree_cumulative_cost:  sum of est_cost in subtree
        subtree_total_work:       sum of total_work in subtree  
        subtree_max_depth:        depth of deepest leaf below
        is_pipeline_breaker:      1 if blocking op waiting on slow I/O child
        child_selectivity_risk:   max Q-error in subtree
    """
    if not operators:
        return operators
    
    n = len(operators)
    
    # Initialize per-node aggregated values
    io_risk = [0.0] * n        # I/O risk score for this node
    sel_risk = [0.0] * n       # Selectivity risk for this node
    subtree_io = [0.0] * n     # Max I/O risk in subtree
    subtree_cost = [0.0] * n   # Cumulative cost in subtree
    subtree_work = [0.0] * n   # Cumulative work in subtree
    subtree_depth = [0] * n    # Max depth in subtree
    subtree_sel = [0.0] * n    # Max selectivity risk in subtree
    pipeline_break = [0] * n   # Is pipeline breaker waiting on I/O child
    
    # Step 1: Compute per-node scores
    for i, op in enumerate(operators):
        io_risk[i] = compute_io_risk_score(op)
        sel_risk[i] = compute_selectivity_risk(op)
        subtree_io[i] = io_risk[i]
        subtree_cost[i] = op.get('est_cost', 0.0)
        subtree_work[i] = op.get('total_work', 0.0)
        subtree_depth[i] = op.get('depth', 0)
        subtree_sel[i] = sel_risk[i]
    
    # Step 2: Post-order traversal (process children before parents)
    # Since operators are in DFS order, we process in reverse
    for i in reversed(range(n)):
        op = operators[i]
        children = op.get('children_indices', [])
        
        if not children:
            continue  # Leaf node — already initialized
        
        # Aggregate from children
        child_max_io = 0.0
        child_total_cost = 0.0
        child_total_work = 0.0
        child_max_depth = 0
        child_max_sel = 0.0
        
        for ci in children:
            if ci < n:
                child_max_io = max(child_max_io, subtree_io[ci])
                child_total_cost += subtree_cost[ci]
                child_total_work += subtree_work[ci]
                child_max_depth = max(child_max_depth, subtree_depth[ci])
                child_max_sel = max(child_max_sel, subtree_sel[ci])
        
        # Propagate up: parent gets max/sum of children's subtree values
        subtree_io[i] = max(io_risk[i], child_max_io)
        subtree_cost[i] = op.get('est_cost', 0.0) + child_total_cost
        subtree_work[i] = op.get('total_work', 0.0) + child_total_work
        subtree_depth[i] = max(op.get('depth', 0), child_max_depth)
        subtree_sel[i] = max(sel_risk[i], child_max_sel)
        
        # Pipeline breaker detection:
        # If THIS operator is a blocking op (Sort, Hash Join, etc.)
        # AND any child has significant I/O risk → pipeline stall likely
        op_type = op.get('op_type', '')
        if op_type in BLOCKING_OPS and child_max_io > 2.0:
            pipeline_break[i] = 1
    
    # Step 3: Write back to operator dicts
    for i, op in enumerate(operators):
        op['child_max_io_bottleneck'] = round(subtree_io[i], 4)
        op['subtree_cumulative_cost'] = round(subtree_cost[i], 4)
        op['subtree_total_work'] = round(subtree_work[i], 4)
        op['subtree_max_depth'] = subtree_depth[i]
        op['is_pipeline_breaker'] = pipeline_break[i]
        op['child_selectivity_risk'] = round(subtree_sel[i], 4)
        # Also store the node's own I/O risk (useful as a standalone feature)
        op['self_io_risk'] = round(io_risk[i], 4)
    
    return operators


def propagate_tree_features_from_df(df_ops, group_keys=('env_id', 'query_id', 'hint_set', 'hint_variant')):
    """
    Apply tree feature propagation to a DataFrame of operators.
    
    Groups operators by (env_id, query_id, hint_set, hint_variant) to reconstruct
    individual query plan trees, then propagates features within each tree.
    
    Args:
        df_ops: DataFrame with operator-level data
        group_keys: columns to group by to isolate individual plan trees
    
    Returns:
        DataFrame with new structural columns added
    """
    import pandas as pd
    
    logger.info("Propagating tree-structural features (post-order traversal)...")
    
    new_cols = ['child_max_io_bottleneck', 'subtree_cumulative_cost',
                'subtree_total_work', 'subtree_max_depth',
                'is_pipeline_breaker', 'child_selectivity_risk', 'self_io_risk']
    
    for col in new_cols:
        df_ops[col] = 0.0
    
    n_trees = 0
    
    for group_key, group in df_ops.groupby(list(group_keys)):
        # Sort by tree_position to maintain DFS order
        group = group.sort_values('tree_position')
        ops_list = group.to_dict('records')
        
        # Reconstruct children_indices if not present
        # (parquet serialization may lose list fields)
        pos_to_idx = {op['tree_position']: i for i, op in enumerate(ops_list)}
        for op in ops_list:
            if 'children_indices' not in op or not isinstance(op.get('children_indices'), list):
                op['children_indices'] = []
        
        # Rebuild parent-child relationships from depth/position
        # (more robust than relying on serialized children_indices)
        for j in range(len(ops_list)):
            ops_list[j]['children_indices'] = []
        
        for j in range(1, len(ops_list)):
            parent_idx = ops_list[j].get('parent_idx', -1)
            if parent_idx == -1:
                # Find parent by depth: closest preceding node with depth-1
                target_depth = ops_list[j].get('depth', 0) - 1
                for k in range(j - 1, -1, -1):
                    if ops_list[k].get('depth', 0) == target_depth:
                        parent_idx = k
                        break
            if 0 <= parent_idx < len(ops_list):
                ops_list[parent_idx]['children_indices'].append(j)
        
        # Run propagation
        propagate_tree_features(ops_list)
        
        # Write back to DataFrame
        for i, op in enumerate(ops_list):
            df_idx = group.index[i] if i < len(group.index) else None
            if df_idx is not None:
                for col in new_cols:
                    df_ops.at[df_idx, col] = op.get(col, 0.0)
        
        n_trees += 1
    
    logger.info(f"  Tree propagation complete: {n_trees} trees processed, "
                 f"{len(new_cols)} new structural features added")
    
    return df_ops


# ═══════════════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("HALO v4 — Bottom-Up Tree Feature Propagation")
    print("=" * 60)
    
    # Example: synthetic 3-node tree
    #   0: Hash Join
    #     1: Table Scan (lineitem, 180M rows, high I/O)
    #     2: Index Lookup (orders, low I/O)
    
    test_ops = [
        {
            'tree_position': 0, 'op_type': 'HASH_JOIN', 'depth': 0,
            'est_cost': 100.0, 'total_work': 5.0, 'loops': 1,
            'actual_rows': 1000, 'self_time': 5.0, 'row_est_error': 0.2,
            'children_indices': [1, 2], 'parent_idx': -1,
        },
        {
            'tree_position': 1, 'op_type': 'TABLE_SCAN', 'depth': 1,
            'est_cost': 50000.0, 'total_work': 200.0, 'loops': 1,
            'actual_rows': 180000000, 'self_time': 200.0, 'row_est_error': 0.5,
            'children_indices': [], 'parent_idx': 0,
        },
        {
            'tree_position': 2, 'op_type': 'INDEX_LOOKUP', 'depth': 1,
            'est_cost': 10.0, 'total_work': 0.5, 'loops': 1000,
            'actual_rows': 1000, 'self_time': 0.0005, 'row_est_error': 0.1,
            'children_indices': [], 'parent_idx': 0,
        },
    ]
    
    propagate_tree_features(test_ops)
    
    print("\nAfter propagation:")
    for op in test_ops:
        print(f"  [{op['tree_position']}] {op['op_type']}:")
        print(f"    child_max_io_bottleneck = {op['child_max_io_bottleneck']:.3f}")
        print(f"    subtree_cumulative_cost = {op['subtree_cumulative_cost']:.1f}")
        print(f"    subtree_total_work      = {op['subtree_total_work']:.1f}")
        print(f"    is_pipeline_breaker     = {op['is_pipeline_breaker']}")
        print(f"    child_selectivity_risk  = {op['child_selectivity_risk']:.3f}")
        print(f"    self_io_risk            = {op['self_io_risk']:.3f}")
    
    print("\n✅ Tree propagation module ready.")
