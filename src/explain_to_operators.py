#!/usr/bin/env python3
"""
EXPLAIN to HALO Operator Extractor
Converts MySQL EXPLAIN FORMAT=JSON output for STATS benchmark into
HALO's unified operator dictionary format.
"""
import json
import logging

def parse_explain_json(explain_json: str, query_id: str, env_id: str = 'A_NVMe') -> list:
    """Parses JSON output from EXPLAIN FORMAT=JSON into a list of HALO operator dicts."""
    try:
        plan = json.loads(explain_json)
        if 'query_block' in plan:
            plan = plan['query_block']
        
        ops = []
        _traverse_plan(plan, ops, depth=0)
        
        # Post-process list to add HALO expected fields
        for idx, op in enumerate(ops):
            op['query_id'] = query_id
            op['env_id'] = env_id
            op['benchmark'] = 'STATS'
            op['operator_id'] = f"{query_id}_{idx}"
            
            # Since we have no actuals, plug in estimates as fallback
            # (row_est_error = 0 assumption)
            if 'actual_rows' not in op:
                op['actual_rows'] = op['est_rows']
            
            # Infer 'self_time' and 'total_work' roughly if missing
            op['self_time'] = op['self_time'] if 'self_time' in op else (op['est_cost'] * 0.001)
            op['total_work'] = op['total_work'] if 'total_work' in op else op['est_cost']
            op['loops'] = 1  
            op['row_est_error'] = 0.0
            
        return ops
    except Exception as e:
        logging.getLogger().error(f"Error parsing EXPLAIN for {query_id}: {e}")
        return []

def _traverse_plan(node, ops_list, depth):
    if not isinstance(node, dict):
        return

    # Handle nesting constructs like nested_loop
    if 'nested_loop' in node:
        for child in node['nested_loop']:
            # The nested_loop node itself represents the join operation
            op = _create_base_op(child, depth)
            
            # Determine Join Type
            if 'table' in child:
                tab = child['table']
                if tab.get('access_type') in ('eq_ref', 'ref'):
                    op['op_type'] = 'NL_INNER_JOIN'
                else:
                    op['op_type'] = 'HASH_JOIN' # Fallback for unindexed join
            else:
                 op['op_type'] = 'NL_INNER_JOIN'
                 
            ops_list.append(op)
            _traverse_plan(child, ops_list, depth + 1)
        return

    if 'table' in node:
        t = node['table']
        op = _create_base_op(t, depth)
        
        # Access types
        atype = t.get('access_type', 'ALL').upper()
        if atype == 'ALL':
            op['op_type'] = 'TABLE_SCAN'
        elif atype in ('REF', 'EQ_REF'):
            op['op_type'] = 'INDEX_LOOKUP'
        elif atype in ('INDEX', 'RANGE'):
            op['op_type'] = 'INDEX_SCAN'
        else:
            op['op_type'] = 'FILTER'

        op['table_alias'] = t.get('table_name', '')
        ops_list.append(op)
        return
        
    if 'grouping_operation' in node or 'ordering_operation' in node:
        op = {
            'op_type': 'AGGREGATE' if 'grouping_operation' in node else 'SORT',
            'depth': depth,
            'est_rows': 100, # default
            'est_cost': 5.0, # default 
            'table_alias': ''
        }
        
        child_node = node.get('grouping_operation') or node.get('ordering_operation')
        
        if isinstance(child_node, dict):
            if 'cost_info' in child_node:
                 op['est_rows'] = float(child_node['cost_info'].get('estimated_rows', op['est_rows']))
                 op['est_cost'] = float(child_node['cost_info'].get('query_cost', op['est_cost']))
        
        ops_list.append(op)
        _traverse_plan(child_node, ops_list, depth + 1)
        return

def _create_base_op(node, depth):
    op = {
        'depth': depth,
        'table_alias': '',
        'est_rows': 1.0,
        'est_cost': 1.0
    }
    
    # In nested loops, 'cost_info' is under the node. If it's a table node, it's under 'table'.
    target = node.get('table', node)
    
    if 'cost_info' in target:
        c = target['cost_info']
        prefix_cost = float(c.get('prefix_cost', 0.0))
        eval_cost   = float(c.get('eval_cost', 0.0))
        read_cost   = float(c.get('read_cost', 0.0))
        
        # We roughly guess the operator's OWN cost
        op['est_cost'] = float(c.get('query_cost', prefix_cost))
        
    if 'rows_produced_per_join' in target:
        op['est_rows'] = float(target['rows_produced_per_join'])
    elif 'rows_examined_per_scan' in target:
        op['est_rows'] = float(target['rows_examined_per_scan'])
    elif 'cost_info' in target and 'estimated_rows' in target['cost_info']:
        op['est_rows'] = float(target['cost_info']['estimated_rows'])
        
    op['est_rows'] = max(1.0, op.get('est_rows', 1.0))
    op['est_cost'] = max(0.01, op.get('est_cost', 1.0))
    return op
